# LG6-SiLU-PB：去掉空间门控的RGB旁路独立训练对照

2026-10-16，分支research/lg6-silu-plain-bypass。用户批准补充对照。现有A=LG6-SiLU，现有GB=A+空间门控RGB旁路；本候选PB=A+相同RGB旁路，不训练B0/A/GB。

公式：R=Conv3x3(3→48,padding1,biasTrue)(I)，Y=PixelShuffle4(T_A+R)。I为原RGB[0,1]输入，不经减均值乘255；T_A原A的48相位重建输出。删除整条image_gate卷积、sigmoid及乘法，不使用固定参数alpha。只删28个门控参数，其它A主干、SiLU、六局部线性门控、原四路融合/重建头以及旁路卷积保持。

尺寸I N×3×H×W，R/T_A N×48×H×W，相加后shuffle输出N×3×4H×4W；HR256对应LR64。RGB每颜色16个相位，无广播、不改变第六块第一层激活返回特征。R可有符号，输出内不clip，评测量化/裁边/Y通道不改。

R.weight/bias=0，初始输出等于A。先构建A，再隔离随机流创建R，共有初始参数、数据随机流相同。只从零独立训练50k，不载入已训练GB权重旁路后续训。固定任意相同共有权重、将GB门控设为1时，PB与GB输出相同，此恒等关系已工程验证；独立训练后的结果不是单checkpoint推理消融。

参数：PB516088，GB516116，相差28；PB相对A514744新增1344，相对B0498208新增17880。服务器manifest extra_deployed_parameters相对B0，不误解为相对A。无门控旁路MACs1296HW（LR256约84.93M），不含加法/访存；延迟尚未实测。

训练完全沿用seed10、DF2K3450、HR256、有效batch64/micro16、RGBL1、Adam2e-4 betas(.9,.999) eps1e-8、原MultiStep[200k,400k,600k,800k]gamma.5，50k内常数LR，FP32、npy_mmap、eval/save1000。调用未修改原trainer/sampler/metrics。相同最高DIV2K_valid PSNR选权重（平分最早），单权重报告全部公共集；补充各数据集峰值并附权重，不跨checkpoint求平均。

参考A完整服务器备份/home/featurize/work/span-x4-research/records/FEAT-LG6-SILU-50K-01-1792052706657616355。server.py核对共同配置（含local_gate）、环境和真实数据哈希。与GB的对比引用仓库已归档50k报告research/docs/module-study/results/FEAT-LG6-GB-50K-01/comparison-50000.json，不要求重新训练或加载GB权重；结果包含selected_metrics_delta_vs_gated和peak_psnr_delta_vs_gated。

prepare.py生成单候选LG6-SiLU-PB，独立logs/plain_bypass/<group>；server.py支持nohup调用，训练成功、work完整备份SHA256校验、曲线生成后，显式--auto-release才释放实例；检查/训练/备份失败不释放。共用本机GPU队列锁。备份含权重、raw loss、逐图指标、config、比较报告、单张三联图curves/seed-10/LG6-SiLU-PB-overview.png。数据/缓存不备份。

CPU/CUDA工程verify包括初始A恒等、共有参数/RNG、RGB相位、GB固定G1严格一致、首步旁路学习、20步有限梯度、严格重载续一步、部署计数。integration.py检查真实入口累积/缓存/断点优化器scheduler/RNG与连续训练一致、JSON恢复和评测频率。合成验证不代表真实PSNR或长训稳定；服务器需运行CUDA verify。

判定：PB≈GB或PB>GB时，空间门控独立收益证据不足，优先简单版本；GB>PB且幅度与逐图结果一致时，才增强门控增量证据。单seed不声称统计显著/跨seed稳健；不事后因一个数据集负向就更改主要规则。暂不叠模块、不修改LR/损失。本轮未启动真实服务器训练。
