# 深化旁路：文献筛选与非线性多尺度设计讨论稿

2026-10-16。用户授权研究讨论，当前不修改模型、不启动训练。既有GB/PB均保留，不因为本设计预期涨分而替换真实最优结果。

## 已有证据和问题

GB最高DIV2K30.234560232，PB30.223241852，门控仅在部分集合小幅有利；直接图像旁路本身在seed10/50k下有正向证据。CPU诊断显示G约1.063、弱空间变化；旁路幅值接近输入。由此优先研究旁路内容建模，暂不增加门控复杂度。原主干已有较大感受野，不能把旁路3x3写成整个模型仅3x3或认定它是已证实瓶颈。

## 文献与核验范围

1. SMFANet，ECCV2024（主会，优先近三年依据）。完整作者论文 https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf 已读取§3.2：局部细节与更广上下文的分支结合，并包含非线性处理。它支撑“互补上下文与局部细节值得共同建模”的动机，不证明下面RGB旁路或3x3/5x5在SPAN上有效，也不是直接复现SMFA的下采样/方差/非局部分支。
2. SAFMN，ICCV2023。官方论文PDF https://openaccess.thecvf.com/content/ICCV2023/papers/Sun_Spatially-Adaptive_Feature_Modulation_for_Efficient_Image_Super-Resolution_ICCV_2023_paper.pdf 。其多尺度调制作为补充背景；2023已处近三年边界外，不作为唯一近期支撑。
3. NTIRE2026 ESR报告§4.1.4 https://arxiv.org/html/2604.03198v1#S4.SS1.SSS4：图像近像素旁路作为结合项目结果的旁路依据。明确是workshop挑战报告，不冒称主会消融证实本模块。
4. Multi-scale feature fusion distilled attention network for efficient image super-resolution，Applied Soft Computing2025，DOI10.1016/j.asoc.2025.113382：仅检索到出版商题录/摘要，正文403，尚不用于细节迁移或分区断言。
5. A Taylor expansion-based texture and edge-preserving interpolation approach for arbitrary-scale image super-resolution，Pattern Recognition2025，DOI10.1016/j.patcog.2025.111965：仅题录/摘要，正文403，且任意倍率任务不同。不据此主张本项目固定x4梯度旁路已获支撑。不写未经核实的年度/学科期刊分区。

边缘检索中的guided thermal SR、扩散感知SR、3D点云/视频SR与本项目证据范围不一致，不直接移植其结论。2022 EFDN超出优先年限，暂不作为新方向主依据。当前优先多尺度，边缘不是被证明无效，而是缺少足够匹配且已核全文的近期依据与本项目机制证据。

## 必须避免的伪增量

并行Conv3x3(I)+Conv5x5(I)若纯线性，可将3x3补零嵌入5x5并合并bias；等价于一个5x5卷积，不能仅凭两分支名称宣称自适应多尺度融合。

Sobel/梯度来自同一LR图像，没有新增观测。固定线性梯度后接线性1x1，原普通3x3卷积即可表达相应映射；后接更大卷积/非线性可能改变能力，但需按实际有效核/非线性和初始化分析。不能把LR梯度当成真实HR细节或自动新增信息。

## 推荐候选：保持G，只升级R（讨论稿）

A原52通道六SPAB门控+SiLU、四路融合、重建头完全不动；保留当前GB的RGB3x3→1、2sigmoid门控G，不同时更改其输入/形式。

F=SiLU(Conv3x3[3→16](I))。

U3=SiLU(DWConv3x3[16→16,groups16](F))。

U5=SiLU(DWConv5x5[16→16,groups16](F))。

U=Concat(U3,U5)，R=P1x1[32→48](U)。

G=2sigmoid(Conv3x3[3→1](I))，Y=PixelShuffle4(T_A+G*R)。

I原RGB[0,1]，不减均值、不乘255；主干仍走原处理。stem stride1 padding1；DW3 padding1、DW5 padding2；无pool、插值、空洞、BN/LN或softmax。各分支共享LR网格。两分支非线性意味着通常不能折成一个固定线性5x5。此网络仍是局部，不叫全局/nonlocal；相对RGB的分支有效感受野为5x5、7x7，不是整个网络感受野。

尺寸：I N×3×H×W；F/U3/U5 N×16×H×W；U N×32×H×W；R/T_A N×48×H×W；G N×1×H×W，只有48通道广播；shuffle输出N×3×4H×4W。输出投影无激活，保留有符号修正。HR256训练对应LR64。原block6第一层激活特征的返回/融合语义不变。

## 初始化及启动

先按原随机流完整构建A，再隔离RNG创建旁路。stem/DW正常随机初始化，不全零；P.weight/bias=0；G前卷积weight/bias=0，使R=0、G=1，初始输出等于A，也等于旧GB初始映射。不是载入训练完GB权重热启动。

P首步可学习，stem/DW与G首步因P/R为0暂时无梯度；之后P更新后必须检查各层梯度和真实更新。若所有层全零，多个路径可能无法正常启动；禁止再加零初始化alpha。恒等起点不保证长训稳定，也不证明该初始化最优。

## 参数及算子成本（所有卷积含bias）

stem：3*16*9+16=448；DW3：16*9+16=160；DW5：16*25+16=416；P：32*48+48=1584；G：3*1*9+1=28。旁路合计2636。相对A514744总参数517380；比旧GB516116多1264约0.245%。这些不是等容量对照，不可将任何提升全部归因于多尺度。

旁路卷积MACs=(432+144+400+1536+27)*H*W=2539HW，LR256约166.40M；旧GB为1323HW约86.70M。新增激活/分支中间张量也增加访存；参数小不保证延迟小，需统一设备实测。

## 实验建议及机制边界

若用户确认实现，第一轮只新增这个组合候选，从零seed10、同DF2K、HR256、有效batch64/micro16、Adam2e-4、原MultiStep、L1、FP32、50k、eval1000。不改loss/EMA/LR，不重训旧组。主要比较旧GB（既有验证规则主结果），同时展示PB，不能挑较弱基线；同预算验证选单一权重，并另列各数据集峰值。

第一轮只能证明“非线性多尺度旁路整体”是否优于旧旁路，不能独立归因于多尺度、非线性、隐藏宽度或参数量。若明显有价值且需要论文机制证据，再考虑等宽单尺度或等容量对照，不一开始强制支付全部消融。若失败按预算停止，不无限叠加。

当前：文献筛选与具体讨论稿已完成；未创建新模型/配置，未启动训练。预期收益是待验证假设，不承诺PSNR上涨。
