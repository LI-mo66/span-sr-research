# SG6 + ASSM 中段原型：工程研究阶段

日期：2026-10-21。论文：MambaIRv2, CVPR 2025。

固定基座：`e731483d4ce4e325f3d64dbbc80add342b46f7f1`。
独立分支：`research/sg6-assm-mid`；独立目录：`research/sg6_assm/`。
仅此新目录与基座不同。SG6、原始 SPAN、公共训练器和历史实验不修改。

**状态：小尺寸 CPU 与 GPU 参考计算通过；原生 CUDA 内核未验证；尚无真实数据 PSNR。**
本目录没有自动启动 50K 的入口。先完成内核、实际 micro16 显存、整图尺寸和公共训练器集成闸门，
再制作唯一的新候选 50K 任务，不能把参考计算当生产训练后端。

## 依据和假设

SG6 的六个 DW3x3 空间修正属于局部建模。前一轮 SG6+ATD 尾部注意力在同 seed/50K 下
六集合 PSNR 均未超过 SG6；其对比原文保存在 `history/sg6-atd-comparison-50000.json`。
这不能证明尾部位置或归一化是失败原因，也不能证明所有全局方法无效。
其服务器备份为
`/home/featurize/work/span-x4-research/records/FEAT-SG6-ATD-LR2E4-50K-01-1792536073170495223`。

新假设：在局部处理中段引入语义排序后的递归状态聚合，随后由后三块继续处理，可能更适配
SG6 的重建路径。此处“语义”是无标签的可学习特征类别，不是预训练语义或额外观测数据。
所有信息仍来自同一幅 LR 图像；增加的是聚合能力，不是外部高频真值。

论文及原码：
- https://openaccess.thecvf.com/content/CVPR2025/html/Guo_MambaIRv2_Attentive_State_Space_Restoration_CVPR_2025_paper.html
- https://arxiv.org/html/2411.15269v2
- https://github.com/csguoh/MambaIR/tree/33d7b3460c4665229334cc8de38c7f4c766ed3be

只借鉴 ASSM 的 SGN/ASE 核心，不移植窗口注意力和两个 FFN，不声称完整 MambaIRv2 复现。
原型的修改及许可证逐项列于 `vendor/NOTICE.md`。

## 衔接及为什么不机械增加激活/卷积

```text
原预处理 -> conv_1 -> SG6 block1 -> block2 -> block3
  -> F + channel_scale * ASSM(LN(F))
  -> 原 block4 -> block5 -> block6 -> 原 conv_2/concat/conv_cat
  -> 原上采样卷积 -> 加入原 GB 图像旁路 -> PixelShuffle
```

`BlockWithASSM` 包裹原 block3，不重写整网 forward。
block3 的 `(output, first, gate)` 返回契约保留；只修正供 block4 使用的 output。
原 SG6 中该位置的 first/gate 不直接送入融合头；block6 的 first 仍沿原路径进入融合头。
浅层 feature、block1 跳连、原融合卷积与 RGB 旁路均保持原样。

ASSM 内部为：
`LN(F) -> 1x1投影 -> x*sigmoid(DW3x3(x)) -> 按类别排序 -> 状态扫描
-> LN -> Linear52到52 -> 逆排序 -> scale -> 加回F`。
路由由同一份 LN(F) 产生，使用 Linear52到17 -> GELU -> Linear17到64。

已有 1x1 输入投影、输出 Linear 和 CPE；再添同尺寸卷积不是对齐的必要条件。
已有路由 GELU 和 CPE sigmoid；外层相加后不再套激活，保留原 SG6 带符号的直通特征。
官方 ASSM 中定义但没有调用的 SiLU 不被当作“已使用模块”宣传。
分支 LayerNorm 只规范新模块的输入；主路 F 不被替换成 LN(F)。
初始通道 scale=0.01 是保守、非零的设计起点，不是已证最优；输出投影不置零。
因此没有外层零门控与内层零投影同时阻断学习的问题。

## 接口表

|项目|约束|
|---|---|
|输入/输出|N,52,H,W；不降采样、不插值、不跨图拼接|
|token 坐标|N,HW,52，按行展开，扫描后同一逆置换还原|
|CPE 位置|在原 H,W 网格计算 DW3x3，不能在语义排序后假装恢复空间网格|
|归一化|每像素 52 通道 LayerNorm，不跨空间或 batch 求统计量|
|prompt 对齐|值特征与对应像素 prompt 使用同一个排序索引|
|形状边界|无窗口整除约束；1x1、奇数、不等边、batch1/2 已测|
|dtype/device|FP32；生产 scan 必须 CUDA，同 device，无混合精度|
|状态|每图、每次 forward 从零起始；没有跨图状态、EMA 或持久提示更新|
|梯度|主路直通；新路全接最终 RGB L1；离散排序无梯度，prompt 用 ST softmax 梯度|
|残差|ASSMCore 返回修正特征，不含外层 F；只做一次 F+scale*delta|

单次语义排序扫描不等于每个像素都读取全部其他像素的当前值。排序能使远处相似像素在序列
邻近；已扫描状态用于后续读出。提示池是学习参数，不能冒称它就是当前图像的全像素池化。

## 两处必须显式处理的官方实现差异

锁定官方源码的 ASSM：对值特征排序后，直接向扫描传入未排序 prompt。
本变体将两者共同排序，确保每个扫描位置的读出提示属于该位置的像素。
这是源码观察与本候选的语义选择，不称为作者已确认的 bug。

官方 forward 在 train/eval 都调用 Gumbel sampling。本变体采用无噪声 ST hard softmax：
训练 forward 与评测都为 argmax；训练 backward 保留 softmax 代理梯度。
同类别用稳定排序保留原行序；重复评测不消耗训练 RNG。
这些是有意修改，故不能宣称将获得论文原模块的相同结果。

## 固定配置与成本

官方轻量 x4 配置采用 C48、state8、expand1、tokens64、rank32；原文配置保存在 vendor。
本原型仅将 C 调整到 SG6 的52，因此 dt_rank=ceil(52/16)=4。
没有凭通道比例随意缩减状态、提示池或改变训练 LR。

新增参数 **12,417**，SG6 部署 **535,772**，合计 **548,189**，增加约 **2.32%**。
公共元数据的累计 B0-extra 为49,981，不能误当作相对SG6的新增量。
没有实测原生 kernel 速度、micro16 显存和 FLOPs，不凭参数少声称更快。

拟沿用 SG6：seed10、从零、50K cap、LR2e-4、原1M MultiStep时间轴、micro16累积4、
HR256、RGB L1、每1K评测、Y/uint8/shave4/整图/无自集成；按 DIV2K_valid 选一个权重报告所有集。
不采用论文原训练配置的 HR192/betas0.99 等设置，避免破坏 SG6 对照。

## 已测证据及未过闸门

`engineering-cpu.json`、`engineering-cuda-reference.json`：
- 固定 SG6 初始化及初始化后 RNG 精确保留；新增模块使用独立 seed60011。
- 同时排序提示和值、逆置换、原码方程适配对照的前向及梯度检查通过。
- 保留非零 GB 图像旁路与六个空间修正；关闭新模块精确还原 SG6。
- 最终 RGB L1 下全部22个新增参数张量真实更新，六个原 spatial depth 分支能启动。
- optimizer 无重复遗漏、严格保存重载和两步恢复后权重一致。
- 同形状 batch 改变另一幅图，不影响第一幅输出；重复 eval 完全一致且不改变 RNG。
- 改变 batch 大小时，原 SG6 与本模型均存在 cuDNN/TF32 舍入差异。原型将其记录为诊断，
  不把这种本已存在的差异误称为跨图信息泄漏，也不放宽输出误差门槛掩盖它。

参考计算最多4096 tokens，仅用于小诊断，代码拒绝静默用于真实50K。
本地未安装 mamba_ssm，原生内核未通过；见 `local-kernel-probe.json`。
`probe_cuda.py` 只查询现有 ABI/依赖并测试已有内核：不安装、不升级、不训练、不关机。
若依赖缺失，下一步按服务器实际 torch/CUDA/ABI 配置在独立环境解决，保护原SG6环境。

只有原生内核前向/反向、重复性、实际训练 batch/最大整图显存、公共训练器集成、SG6档案
复用审计均通过，才交付 nohup 50K 脚本。正式训练及备份后自动释放沿用用户已有要求；
工程检查不触发自动释放。

下一候选仍待后续筛选；本实验结束后须提醒，不因一轮失败叠加多个模块。
