# SG6上的UCAN双分支上下文与Q/K跨块传递

状态：v2独立实验工程实现；已补服务器交付入口，本地未启动真实50K训练。
工程结果见 `ENGINEERING.zh.md`；服务器使用说明见 `SERVER.zh.md`。

固定基座：`research/gb-spatial-gate6`，commit
`e731483d4ce4e325f3d64dbbc80add342b46f7f1`，部署535772参数。
独立本地分支：`research/sg6-ucan-qk-pair`。本分支直接从固定commit创建。
所有新增文件仅在 `research/ucan_qk_pair/`，不修改原SPAN、SG6、共享trainer、
loss、优化器、数据或评测源码。来源固定版本与改动见 `SOURCES.md`。

## 可证伪问题和顺序

SG6没有显式跨块Q/K传递，但这不是已证明的性能瓶颈。
先检验两处独立双分支上下文是否超过SG6，再检验复用旧Q/K处理当前V是否值得。
已知SGLA单点负结果不被抹去，也不被概括为所有全局方法无效。

```
block1 -> block2 -> context2 -> block3 -> block4 -> context4 -> block5 -> block6
                        |                           ^
                        +-- raw Q2/K2 (reuse only) --+
```

原浅层、block1输出、第6块内部first特征、conv2、conv_cat、RGB门控旁路和
PixelShuffle全部保留。上下文仅写回当前块的主输出，不重写SG6块内gate或first。

| 模式 | context2 | context4 | 部署参数 | 真实训练状态 |
|---|---|---|---:|---|
| off | 无 | 无 |535772|仅必要时补同协议对照|
| independent | 当前Q/K/V | 当前Q/K/V |554128|唯一首轮候选|
| reuse | 当前Q/K/V | 旧Q/K、当前V |551372|仅工程验证，入口拒绝真实训练|

新增参数18356/15600，分别为SG6的3.43%/2.91%，均小于SG6的10%。
训练态重参数化卷积与部署融合参数不同，不能拿约2.2M训练参数冒充部署大小。
两版所有共同参数初始化完全一致；复用版仅删掉context4的Q/K投影。

每项真实新实验必须从固定基座独立分支、独立目录/配置/ID开始。
当前只有independent处于首轮候选状态；reuse是同一工程包中的机制测试，
若以后获准做真实训练，必须从固定commit另建分支和独立实验目录，移交
同一版本适配代码及来源记录，不能直接在本checkout开始第二项实验。

## 接口、分布和归一化

| 环节 | 布局/像素/数值约束 |
|---|---|
| 输入输出 | Bx52xHxW，正H/W，不下采样、不补窗口、不裁切 |
| token | Bx(HW)x52，n=h*W+w；H/W交换不是等价布局 |
| 分支LN | 每像素沿52通道，eps1e-5；主残差不归一化；两处独立参数 |
| 投影 | 原始Q/K/V为Bx(HW)x26，单头；Hedgehog将每头26映射到52 |
| 交互 | 空间和通道各26维，concat52，原输出投影52->52 |
| 位置 | 保留源1D展开位置和频率截断；不同图宽的语义差异仍属迁移风险 |
| LePE | 当前V按同一行优先顺序恢复26xHxW，DW3x3零padding1 |
| Q/K状态 | 当前forward内局部变量，不detach、不跨图缓存，验证B/H/W/dtype/device |
| 写回 | x+RMS(x).detach()*correction；RMS逐图沿C/H/W计算，最小1e-6；不增加可学习gate |
| 精度 | 当前协议FP32，不使用AMP；模型支持测试中的FP64/CPU/CUDA迁移 |

LN只缓解输入尺度差异，不保证跨块通道语义一致；直接把LN后修正量相加仍可能
破坏深层小幅特征。因此v2增加逐图RMS单位恢复，残差主路不变、不额外加LN。
后处保留独立V/输出投影和
Q/K特征映射、位置参数、温度，提供适配能力；是否学会仍待实验。
Hedgehog softmax和通道L2归一化是注意力定义的一部分，不能简单计入“堆LN”。

投影零初始化保留同初始化SG6的精确映射。第一步内部梯度零是预期现象；
必须确认后续内部参数开始真实更新。新层构造隔离随机流，不改变原SG6
权重、采样/增强随机流。沿用原Adam，不同时增加warmup/裁剪/分组LR。

v1合成4步后第4处修正/input RMS约7.86--52.14，不能因有限梯度而放行。
固定同一权重和同一局部输入干预显示：改为绝对值和归一化并做RMS写回后，
该比例约0.0118--0.0123。这只是数值定位，不是独立训练消融或PSNR证据。
v2傅里叶系数除以绝对值和（下限1e-4），有符号系数保留；不再声称这部分
与UCAN原公式完全相同。旧signed路径仅用于源码等价验证和回放诊断。

## 工程验证命令

在独立checkout根目录、已有兼容PyTorch环境中运行；不安装或升级用户环境：

```bash
python research/ucan_qk_pair/verify.py --device cpu --output tmp/ucan-pair/check-cpu.json
python research/ucan_qk_pair/verify.py --device cuda --output tmp/ucan-pair/check-cuda.json
python research/ucan_qk_pair/resources.py --device cuda --output tmp/ucan-pair/resources.json
python research/ucan_qk_pair/training_probe.py --output tmp/ucan-pair/training-probe.json
```

`integration.py --output tmp/ucan-pair/integration.json` 要求源码已commit且干净，
通过真实共享入口做4次合成更新，验证梯度累积、npy缓存、评测/权重选择、
损坏日志恢复和精确续训。所有合成文件使用独立ENGINEERING ID，保留现场。
它不等于真实数据拟合或50K稳定性。

## 未来短筛协议（本轮未启动）

seed10，从零50K，HR256，有效batch64/micro16，Adam2e-4/L1。
完整1M原MultiStep时间轴[200K,400K,600K,800K]保持，50K内不降LR。
每1K评测，无自集成，原量化/Y/crop4/SSIM代码不变。
按DIV2K_valid最高PSNR选一个权重，六集合均报告该权重结果。
另保存50K固定末点及完整曲线，但不改变选权规则，不从测试集选超参。

配置保留基座原始total_updates=1M，50K是执行截停，不重写scheduler。
入口强制显式 `--screen --screen-updates 50000`，防止默认20K或误跑1M。
只允许arm对应ID前缀，拒绝覆盖已有run，续训沿用原指纹检查。

未来independent ID：`FEAT-SG6-UCAN-IND-V2-50K-01`。
在服务器核对数据hash、来源/环境/评测指纹及GPU资源后，准备独立路径配置，
再由用户执行 `launch.sh` 后台启动；只运行这一项，不自动排队其它方案。
Codex本轮不远程启动服务器训练。
SG6历史指标由用户给出，不能因本地合成等价就宣称服务器复用条件全部通过。

独立版50K未显示优势则停止本版本，记录“本协议下未显示优势”；不自动延长、
不启动reuse、不叠LKSA。若独立版显示优势，再审查精度/成本和训练波动，决定
共享版是否进入独立实验。共享减少参数不等于延迟更低；若需要归因共享本身，
再补参数预算对照。短筛优势不代表长训必然优势。
