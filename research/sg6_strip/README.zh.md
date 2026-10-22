# SG6 部分通道条带短筛 v1

固定基座 e731483d4ce4e325f3d64dbbc80add342b46f7f1；独立分支 research/sg6-partial-strip-v1。
所有变化限定本目录。GRep 已取消。本轮借鉴 LKMN/LKFMixer 的部分通道条带思路自行适配，不是作者原模块复现；没有使用其预训练权重、CGFN、Adan、EMA 或 FFT loss。
来源快照：Supereeeee/LKMN 0bb42dab173aaa11a7801ae1a182d017be515151；Supereeeee/LKFMixer 9d947f9cc761dbe874adf4b7c4bcd8b507fd1f32。

## 唯一结构变量

只替换 block_3.spatial_gate.depth。其余五块空间修正、18 个主干 Conv3XC、原 SiLU、52→52 project、可学习线性门控、图像旁路、融合及 ×4 上采样保持原实现。

|接口|实现与约束|
|---|---|
|输入/输出|N,52,H,W → N,52,H,W；同 dtype/device、通道顺序和像素位置|
|前13通道|DW 1×K → DW K×1；两层均带 bias，无内部激活|
|后39通道|保留原 DW3×3 初值；拼接恢复原顺序|
|尺寸|奇数 K、stride1、每轴对称零 padding，无缩放/窗口/token 顺序|
|后续|仍为原 SiLU→project；不添加残差、乘法、归一化或持久状态|
|启动|原 project=0；首步内部卷积梯度为0，第二步应启动|
|部署|仅原 Conv3XC 按原工具融合；含 SiLU 空间修正不能整体折叠|

|候选|新增部署参数|总部署参数|总可训练参数|
|---|---:|---:|---:|
|K15|286|536058|2194986|
|K31|702|536474|2195402|

参数增加 13×(2K−8)，卷积权重 MAC 增量为 13×(2K−9)×H×W；未把拆分拼接、bias、激活和内存访问算入，因此不宣称速度不变。
条带扩大邻域但每通道有效二维核为可分离秩1，并非任意 K×K 核；13通道为固定前13个，无学习选通。K31 在 LR64 patch 上的边界占比较大，可能不利于边缘或纹理。

## 特征与优化检查的含义

新条带从同一随机31-tap母方向裁切，逐通道匹配原随机 DW3 核的 Frobenius 能量；横向 bias=0，纵向 bias 复制原值。仅对内部 IID 输入的线性响应能量有可解释控制，不等于自然图像特征分布匹配，不保持已训练 SG6 映射。
构造过程隔离 RNG，其他参数初值及采样随机流保持一致；从零时 project=0，初始完整函数与 SG6 精确一致。
原 SG6 没有 BN/LN/GN 堆叠，本候选也不新增归一化。保留带符号特征及无约束门控；不会预防性加 norm、裁剪、warmup 或分组 LR。
K15/K31 CPU/CUDA 工程结果见 engineering/。测试包括偏置/边界独立逐偏移 oracle、1×1/奇数/非方形/batch2、梯度启动、实际 Adam 更新、严格保存恢复、FP64 train/eval 与 FP32 部署导出。
本机 torch2.9.1/cu128/4060 不能替代目标 torch2.2.2/cu121/4090 检查；服务器会重跑。

features.py 只在固定 SG6 的4个真实图像中心 LR64 crop 上做尺度压力测试，绝不把临时换层结果当候选性能。
run.py 的稀疏观察器记录前8步及每1K步最后一个 microbatch 的 F、depth、local/spatial 门控幅值、梯度和真实参数更新。观察不改 loss/更新公式/RNG；独立检查观察开关后的下一步参数精确相等。
零投影首步应为0、后续不能长期不动；幅值大、小或负值本身不是失败判据，要联系 loss/PSNR/实际更新解释。真实分布随训练变化，无法在训练前保证匹配。

## 公平协议与历史教训

从零 seed10；L1，Adam2e-4、betas(.9,.999)、eps1e-8；HR256、micro16、有效64；原数据顺序/采样/缓存后端和依赖。
每1K更新记为一个 epoch，50个记录周期=50K；学习率仍按完整1M时间轴200/400/600/800K减半，短筛全程2e-4。
每1K同原评测；DIV2K_valid最高PSNR选一个权重，平局取最早，同权重报告全部集合，不拼公共集峰值。
prepare.py 要求固定SG6完整备份、相同环境/依赖、所有训练及验证输入有序哈希、原50K曲线和48K所选指标匹配，才复用直接对照。
config 中 total_updates=1M 是 scheduler/协议定义；队列强制 --screen-updates 50000，不会自动训练1M。长训须重新讨论并安排同预算 SG6。

沿用已核查的18分支/81次可达研究提交教训：TAB/FFN/SG6+ATD负例不证明所有注意力无效；本轮不再整套叠模块；EASA/MS提醒不要重复空间调制/旁路；ASSM提醒非零梯度不等于有效更新；不靠删LN、改LR掩盖问题；KD的额外教师成本本轮避免。历史来源及细节见 delivery/history-lessons.zh.md。
SG6 旧审计的 TF32 FP32 train/eval 差异仍需区分执行精度与结构等价；本包用FP64检验结构，不放宽旧检查后宣称其通过。正式训练保留归档 TF32 设置。

## 一次启动两个方案

交付包 INSTALL.sh 默认创建新的 /home/featurize/work/span-sg6-strip-v1 checkout。若存在则拒绝覆盖；不会 checkout 或修改 SG6。
设置 PYTHON_BIN=/environment/miniconda3/bin/python3 后运行 bash INSTALL.sh 即在后台串行运行 K15、K31。
可用 REFERENCE_BACKUP、DATA_ROOT、CHECKOUT_DIR、OUTPUT_ROOT、CACHE_ROOT 指定路径，见安装脚本。源文件和 Git bundle 离线提供，不需要网络安装依赖。
默认 CACHE_ROOT=$DATA_ROOT/image_cache（默认 /home/featurize/datasets/image_cache），只复用按图像哈希命名的解码缓存；实验目录/权重完全独立。不存在时在该处创建，原图不修改。可显式指定独立缓存；空间不足时停止，不能静默改 png 后端。
GPU锁覆盖全部预检、两次串行训练及关机请求；已有计算进程则退出，不抢占。两版分别日志/ID/权重/比较/哈希校验备份。

2026-10-22用户明确授权自动关机，取代本候选上一版“保留机器”的默认选择：GitHub拉取后的 launch.sh 默认 AUTO_SHUTDOWN=1，后台 nohup 运行。server.py 本身仅显式传入 --shutdown-after-queue 才启用。原离线包 INSTALL.sh 仍保留不关机入口；本轮请使用 launch.sh。
启动时核查系统 shutdown 可执行文件以及 root/无交互 sudo 权限；无权限则在正式训练前退出。只有队列结算、已有权重与日志备份 SHA256 校验完成、回传ZIP校验完成后才调用 shutdown -h now（非root用 sudo -n）。单版训练失败但备份成功会完成另一版再关机；预检失败、备份/归档失败或结束时检测到其它GPU计算则保留现场。
这是系统关机，不调用 featurize instance release，不主动释放/销毁实例；系统关机是否停止平台计费需按平台确认，本地未验证实际服务器权限或平台关机效果。状态写入组目录 shutdown-status.json。check_power.py 中所有关机均为 mock，没有真正关过任何机器。
AUTO_SHUTDOWN=0 bash research/sg6_strip/launch.sh 可显式保留机器；SHUTDOWN_BIN 可指定系统已有 shutdown 可执行路径。不能保证 SIGKILL、断电或系统崩溃时执行收尾。

预检依次核基座/输入来源、真实trainer两步断点恢复、CUDA工程检查、冻结特征压力探针、真实micro16×4和最大整图资源检查、数据preflight。
临时合成资源测量为SG6及每版各5次Adam更新（2warmup+3计时），不保存权重、不计入正式50K。中位更新时间超过 SG6 的1.10倍则 HOLD_COST；10%是本次节省预算的保守默认线，不是普适统计结论。可在新队列启动前显式改 MAX_TIME_RATIO，续训不能降低已冻结门槛。
正式训练发生非有限 loss/被观察参数梯度/特征/更新则失败停止该版并备份，另一版仍可独立尝试；公共工程预检错误会阻止整个队列，先修复再重开，不能当候选PSNR失败。

queue-status.json 是最终状态。中断正式训练可在源码不变时用 server.py --resume-group 原组路径恢复，仍核指纹；中断预检要使用新组，旧组保留。
比较脚本输出选中权重、50K末点、末3周期均值、逐图差值/胜率/中位数/图像bootstrap区间。图像区间不覆盖训练seed方差。
验证所选增益≤0或末3均值增益≤0，当前版本STOP。正向只进入REVIEW，不自动长训；非常小的正向属于待复核灰区，当前没有跨seed噪声下限，不能宣称显著。检查外测退化、资源和更新日志后再决定。

队列收尾自动生成 SG6-STRIP-RETURN.zip，内含上述诊断、比较、全部逐图指标/曲线/loss/特征更新日志及 SHA256 清单，不含大权重/数据。直接回传该 ZIP 即可；中途失败时先回传队列日志，不用等另一个版本完成。
