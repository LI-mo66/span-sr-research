# SPAN52 六块直接线性门控：设计与实验候选 v1

2026-10-15。独立分支 research/spab-linear-gate6。状态：实现及合成工程验证；没有服务器真实数据训练，没有新PSNR结果。原始 basicsr/、已有 A1-G4 和固定 B0 均不修改。

## 依据、问题与证据边界

NTIRE2026报告 §4.1.2（https://arxiv.org/html/2604.03198v1#S4.SS1.SSS2）给出局部 1×1 线性门控及两次 ReLU，输出 (X+F3)*Conv1x1(F3)。这是 workshop 挑战报告依据，不是该改动在我们 SPAN52 上独立消融涨分的证明。报告整体包含32通道、5块、不同头、图像旁路及训练策略；本方案不称完整复现 SPANV2。报告没有明确此适配的初始化，本方案的0.1偏置由我们提出。

实际原代码门控 sigmoid(F3)-0.5 已可取负值；新增能力是独立可学习的跨通道仿射混合及无界幅值，而非首次允许负值。三层原卷积此前也已混合通道。已有 A1-G4 是最终四路softmax融合，50k未发现一致收益；它与本方案位置、约束不同。

## 精确公式与候选

F1=A(C1(X)); F2=A(C2(F1)); F3=C3(F2); M=P(F3)=WF3+b; Y=(X+F3)*M。

C1/C2/C3 仍是原 Conv3XC，训练及部署重参数化保持原逻辑；P 为带bias、groups=1、stride=1、padding=0的52→52 1×1普通卷积。全程NCHW，不做池化、resize、softmax、sigmoid、归一化、截断或 detach。门控可正可负、无界；(X+F3) 内的 X 也被门控，不存在额外未门控的块外残差。

|候选|激活 A|改变|解释范围|
|---|---|---|---|
|LG6-SiLU|原 inplace SiLU|六块门控替换|门控替换整体效果|
|LG6-ReLU|inplace ReLU|六块门控替换+两次激活替换|组合效果；相对LG6-SiLU可看该门控下激活影响|

不训练两个不同位置、不叠加 A1-G4，不加入旁路、EMA、频域loss、warmup、分组学习率或裁剪。不能凭这两组区分所有门控/激活主效应与交互；没有ReLU-only组时不能说ReLU单独有效。

## 尺寸、语义及衔接

|节点|尺寸|语义|
|---|---|---|
|输入RGB|N×3×H×W|trainer 输入 RGB /255；网络原有减均值、乘255不变|
|原conv_1|N×52×H×W|浅层F0|
|每个SPAB输入X、F1、F2、F3|N×52×H×W|同一LR网格，无额外padding变化|
|门控P(F3)|N×52×H×W|每位置输出52个通道权重，无广播|
|块输出Y|N×52×H×W|送入下一SPAB，六块串联|
|原conv_2(block6输出)|N×52×H×W|最终深层F6|
|原四路concat|N×208×H×W|顺序F0、F6、block1最终输出、block6第一层激活后特征|
|原conv_cat|N×52×H×W|208→52的1×1融合|
|原upsampler卷积|N×48×H×W|52→3×4²，原3×3卷积|
|PixelShuffle×4|N×3×4H×4W|原RGB输出，不额外加均值或输入旁路|

HR256训练裁剪对应LR64：每路52×64×64，concat208×64×64，shuffle前48×64×64，输出3×256×256。支持batch1/2、非正方形、奇数H/W。输入dtype/device继承原FP32 CUDA流程。

SPAB依旧返回 (Y,F1,M)。原out1受 inplace act1影响，必须保留为激活后特征；第六块第二个返回值参与最终融合。ReLU候选有意改变该特征的激活统计，不能解释为只改变门控。未改原网络全局forward，直接沿用原四路融合及头。

## 初始化、梯度和稳定性

先构建完整原SPAN，再隔离RNG构建六个门控；所有共有参数初始值、主训练CPU/CUDA随机流与B0一致。ReLU没有参数且不消耗随机数。P.weight=0，P.bias=0.1，初始M=0.1，各通道、空间一致。不能把W和b都置0，否则块输出为0；也不默认b=1，避免直接放行整块初始幅值。本设置并非B0输出恒等初始化，也并非论文已验证最优设置。

训练初期门控对F3的导数W为0，但输出仍有直接路径 dY/d(X+F3)=0.1。门控参数有梯度 dY/dW 包含 (X+F3)F3，不能因此认为W不会学。必须验证每块weight/bias梯度有限、非零及实际更新；不能再串联一个同为0的可学习scale。

常数0.1不保证后续无界门控稳定：门控和主干可共同放大、前层影响后续块、不同随机种子和数据均可能改变行为。没有在模型中添加硬clamp或偷偷裁梯度。现有trainer非有限loss检查保留；长训前应服务器CUDA工程检查，再真实数据短启动观察loss、输出/门控幅值与梯度，发现异常停止诊断，不从损坏断点续训。本地合成20步不是长训稳定性或PSNR证据。

1×1门控不能简单折进原C3：F3还必须原样供X+F3使用，门控另需其仿射变换。使用原fused_copy只合并Conv3XC，保留六个新门控。

## 资源

每块52×52+52=2756，六块新增16536；部署498208→514744（+3.32%）。只计新卷积，MACs=6×52²×H×W；LR64×64约66.45M，LR256×256约1.063G，不包括加法、乘法及访存，不等于实测延迟。训练态重参数分支参数不等于部署参数；以verify结果核对。

## 固定训练与B0复用

使用已归档B0的DF2K、seed10、HR256、有效batch64/micro16、RGB L1、Adam lr2e-4 betas(0.9,0.999) eps1e-8、MultiStep [200k,400k,600k,800k] gamma0.5、FP32、npy_mmap、eval1000、累计50k；50k内LR不下降。两个候选分别从零训练；没有载入B0已训练权重或A1权重。原trainer、data sampler、metrics原文件不改，由独立entry替换factory；50000作为CLI stop cap直接支持，不修改1M调度时间轴。

B0档案：research/docs/module-study/results/FEAT-MOEFS-G4-LR2E4-20K-01；最佳update48000，验证PSNR30.15989124662389。新结构初始输出不同是实验设计变化，不冒称初始映射与B0相同。公共训练算法/数据/评测代码变更时必须重新核对基线。

最高DIV2K_valid PSNR选权重（同分取最早），报告其全部测试结果；另外保留各数据集峰值及对应checkpoint。不要跨checkpoint或跨数据集平均PSNR。逐数据集平均其图片的指标保持标准口径。

## 准备与调用（本轮未启动真实训练）

先在服务器使用独立checkout，禁止在仍运行的旧训练目录切换版本。代码work，数据/home/featurize/datasets，runs/logs链接高速盘，结果备份work，沿用已确认的服务器规则。训练入口严格要求源码clean，不允许新实验输出污染git status。

```bash
python3 research/linear_gate/verify.py --device cuda --steps 20
python3 research/linear_gate/prepare.py --group FEAT-LG6-50K-01 --data-root /home/featurize/datasets
python3 research/linear_gate/entry.py --config logs/linear_gate/FEAT-LG6-50K-01/LG6-SiLU.json --id FEAT-LG6-50K-01-LG6-SiLU-s10 --preflight
python3 research/linear_gate/entry.py --config logs/linear_gate/FEAT-LG6-50K-01/LG6-SiLU.json --id FEAT-LG6-50K-01-LG6-SiLU-s10 --screen --screen-updates 50000
```

ReLU候选替换config和id中的LG6-SiLU为LG6-ReLU即可。恢复同一run加--resume，严禁改冻结config。上述是同步入口用法，不会nohup、备份或释放；服务器完整后台队列需后续单独配置，不能误称当前已启动。无需重训B0。

验证：verify.py为公式、形状、共有参数/RNG、有限梯度/真实更新、优化器覆盖、严格序列化续一步一致、部署转换；integration.py为真实入口的合成累积、缓存、scheduler/RNG断点恢复、评测频率不扰动训练，使用--variant选择候选。engineering_only只能在显式LINEAR_GATE_ENGINEERING=1的隔离fixture启用，不可用于真实实验逃过数据检查。

## 停止线与后续提醒

先观察真实训练稳定性，在预算内比较固定B0，不因一次负结果自动无限搜索初始化或叠加模块。两候选并列记录，不能只汇报最好的一个。实验结束分析时提醒用户方案2：图像信息旁路；当前不实现，不启动。
