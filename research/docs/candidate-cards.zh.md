# 首轮三个候选：来源、位置、成本、风险

2026-10-11检索；近两年已发表且有官方代码为范围，优先近一年。没有宣称以下机制在SPAN上有效。为保持目标相关性保留TIP/ICLR2025作为近两年备选；2026命中中生成式/真实退化/视频方法不能仅凭更新而优先。

## A：SSIU的MoE-FS思想——原四路融合前自适应门控，优先级1

论文：Structural Similarity-Inspired Unfolding for Lightweight Image Super-Resolution，TIP2025，https://arxiv.org/html/2506.11823v1；官方https://github.com/eezkni/SSIU；锁定deb75d4429e3ef0a244c71476ac2283124becb6c。仓库README称MIT，克隆中未找到独立LICENSE；接入时采用明确标注的独立适配实现，并再次核代码授权，不能把README的声明当完整已核验许可证文件。

代码models/SSUFSR_network.py的MOE用3个1×1产生逐像素/通道权重，在候选分支轴softmax，融合后有投影/残差。52通道原MOE实测11024参数。针对SPAN只借鉴层级选择，保留4路和conv_cat，拟用4个52→52的1×1门控，共11024新增部署参数（2.21%）。初始化logits0，使用4*softmax使原映射不被1/4缩小。空间softmax需要四路logits/权重，须算内存和算子延迟；不要照搬原代码的squeeze操作产生batch广播问题。

位置仅原conv_cat之前，不改变SPAB内部。廉价对照为每路固定可学习常量门控；若常量同样有效，不能宣称图像自适应是关键。失败风险是原conv_cat已足够、门控重复或损失早层信息。MAC新增主要4*C²*H*W，不含softmax/乘法；未运行适配模块，非实测FLOPs。

## B：ESC ConvAttn——部分通道的大核＋动态局部上下文，优先级2

论文ICCV2025，https://openaccess.thecvf.com/content/ICCV2025/html/Lee_Emulating_Self-attention_with_Convolution_for_Efficient_Image_Super-Resolution_ICCV_2025_paper.html；官方https://github.com/dslisleedh/ESC，MIT文件存在；锁定f22c712a57c664360f9af3a77cfc527c414067c4。使用esc/archs/esc_arch.py的ConvolutionalAttention思路，不使用2025年底README明确不建议学术评测的FlashBias更新。

在52通道中的13通道上试13×13静态卷积＋图像条件3×3动态卷积，再52通道1×1混合、残差写回。必须注意原静态大核是通道混合，不是depthwise！预测参数：13²*169=28561；动态生成13→6→117为903；1×1混合2756；52个残差scale；总32272（6.48%）。这是对具体拟实现设计的算术预算，不是已经移植的测量。每LR像素约28561+52²个静态MAC，再加动态图像卷积等，完整延迟必须实测。

位置先block3后与conv_cat后/PixelShuffle前各一个，统一结构尺寸；不同时放多个位置。机制对照为相同静态大核但关闭动态分支并独立训练；参数匹配对照另行补充。原eval源码假定batch1且原地写入x的一部分，适配必须避免污染其他残差路径；静态核共享需要明确注册与状态生命周期。动态生成器末层初始0可以沿用，但外层scale不同时0。失败风险：SPAN原有效感受野已够、动态groups和大核慢、静态部分使实际预算/速度不划算。

## C：AdaIR的频率调制思想——轻量高低频修正，探索优先级3

论文ICLR2025，https://proceedings.iclr.cc/paper_files/paper/2025/hash/fb44a668c2d4bc984e9d6ca261262cbb-Abstract-Conference.html；官方https://github.com/c-yn/AdaIR，MIT文件存在；锁定ccb8b98e49614e07badd0641e5163fa7635c2f02。原任务是多退化图像复原，并非经典bicubic×4超分，这是明确的跨任务迁移风险。

net/model.py的FreRefine是低频生成通道门控、高频生成空间门控，52通道实测3166参数。拟保留调制思路，用固定低通得到L、H=x-L，再做交叉调制和1×1投影，52维小非零scale残差写回；总预测3218参数（0.646%）。低通先固定AvgPool5，廉价同预算对照去掉分频但保留相同投影/门控。此处是受论文启发的独立轻量变体，不是完整AdaIR原模块，不能照搬其整网提升。

完整FreModule52实测43272参数也在10%内，但含FFT与3个cross-attention，开销/任务风险更高，不作为首轮。原fft方法n128和整型mask使64×64LR输入时频带mask可能退化为空，并且取abs会丢符号；未经适配不能直接复制到SPAN。因此首轮不用完整FFT版本。低通/高通并不产生额外观测信息，作用是不同频率处理的归纳偏置。

位置先conv_cat后/PixelShuffle前，第二位置block3后；输出保持LR尺寸。失败风险：低通边界伪影、平坦区变差、分频无收益、现有SPAB已学会同类滤波。必须统计边缘/平坦误差并以完整验证集PSNR/SSIM判断，不能只展示锐利图。

## 排除与待核验记录

- CAFT（Pattern Recognition2026）论文指向https://github.com/Prtick/CAFT，锁定df160b81fc180043b8e08b5e1d6d938709a843e0，当前只有README“will be released soon”，没有模块代码；不满足已开源条件，不进入队列。
- CCNet（DOI10.1016/j.imavis.2026.106189）有官方源码2a40c28365a8260affeb4c62ccec1e5857afb91b，但README仅2字节、未找到LICENSE，Crossref列2026年11月刊期、无published-online字段。当前日期10月11日，在线发表时间不能确认，故暂不按已发表/可复用候选接入。
- FreqOrtho-SR/ASASR等2026真实世界生成式方法与本项目bicubic PSNR/轻量预算不匹配；不因年份新就移植完整系统。
- DLGSANet、SwinIR、PLKSR等较早方法不作为满足本次近两年条件的新增主候选。

上面成本是不同设计的初始预算估计/源模块构造计数，尚没有模块移植精度或速度结果。A+B预测43296新增参数，A+C14242，B+C35490，三者46514都在10%算术预算内，但只会融合单独验证有效的前两种，不能因参数够就堆三个。

## 统一接口约束

| 候选 | 输入/输出 | 写回与状态 |
|---|---|---|
| A |4路N×52×H×W，相同LR像素坐标；权重4×N×52×H×W |softmax只沿4路轴，乘4后concat；无跨batch缓存，不增加重复残差 |
| B |N×52×H×W，13通道处理再恢复52 |空间padding对齐，大核/动态生成器均注册；禁止原地改共享输入；独立修正残差 |
| C |N×52×H×W；L/H空间对齐 |固定低通padding公开，保持有符号H；空间/通道门控不改变布局；无图像间缓存 |

均使用同dtype/device，模块输出不clamp或uint8量化，只有指标阶段量化。检查13×17等奇数非方形输入；逐层幅值、gate、梯度和真实参数变化为工程诊断。训练forward损失联通、eval/reset/checkpoint状态必须严格验证。
