# LG6-SiLU + 空间门控图像旁路（A+B）v1

2026-10-15。分支research/lg6-silu-gated-bypass。仅新增候选LG6-SiLU-GB。A为归档LG6-SiLU整体（六块线性门控+SiLU），不是SiLU单独贡献。B为本文件的空间门控图像旁路。当前没有真实数据新PSNR。

## 精确结构

原RGB输入I∈[0,1]，N×3×H×W。原A主干完整保留：减均值乘255、conv1、六线性门控SPAB、conv2、四路concat208通道、conv_cat208→52、原重建3×3卷积52→48。记其shuffle前输出T_A。

R=Conv3x3(3→48,bias=True,padding1)(I)。

G=2*sigmoid(Conv3x3(3→1,bias=True,padding1)(I))。

Y=PixelShuffle4(T_A+G*R)。

R与T_A均N×48×H×W。G为N×1×H×W，只沿48个RGB/子像素相位通道广播，同一LR像素的全部相位共享空间权重；不跨batch/空间广播，不称HR每子像素独立门控。通道顺序为RGB每颜色16个相位，原shuffle次序保持。输出N×3×4H×4W。LR64对应HR256，任意奇数/非正方形H/W均支持。

旁路不减均值、不乘255；否则与原重建输出数值目标不一致。R无激活，允许负修正；G在数学上(0,2)，有限精度可能到端点。没有softmax、主干互斥竞争或输出内clamp。sigmoid门控仍可能饱和，本设计不声称消除所有饱和问题。G有界不等于R、主干或总输出有界。

## 初始化与学习启动

先完整构建A，再隔离CPU/CUDA RNG构建新层；A共有参数及数据随机流保持。R与G前卷积的weight/bias均0：初始R=0、G=1，故A+B初始train/eval输出严格等于A。不是加载A49k权重，不是初始NN旁路。无需额外可学习alpha，避免多余缩放和双零路径。

首步R卷积有梯度；门控卷积首步因R=0梯度为0，这是明确允许的暂时零梯度。R更新后必须检查G梯度与参数真实更新。零初始化不保证长期稳定；原A中的无界线性门控保持原样，不能把旁路有界G当作整个模型稳定性证明。

两条新卷积都是输入相关学习层，G是空间动态门控而非可吸收的常数alpha。P(RGB)作为RGB直接修正路径不证明严格低频，不能把深浅职责分离当已验证事实。

## 文献与创新边界

NTIRE2026报告§4.1.4 https://arxiv.org/html/2604.03198v1#S4.SS1.SSS4 支持图像直接旁路的动机；其具体方案是NN初始化、深度卷积与另一融合头。本方案的零初始化普通卷积和2sigmoid空间门控是我们的适配假设，并非报告逐项消融验证的模块。A+B>A只支持B在A下的增量，不证明B独立有效或超加性协同，也不自动构成两个论文创新点。

## 资源

R参数48*3*9+48=1344；G参数1*3*9+1=28；新增1372。

A部署514744，A+B部署516116（相对A+0.2665%）；相对原B0新增17908。额外卷积MACs=(1296+27)*H*W=1323HW；LR64约5.42M、LR256约86.70M，不包括sigmoid/加乘/访存。G依赖输入，不能将整条非线性门控旁路折成一个固定卷积。实测延迟/显存尚未知。

## 训练、基线和结果

只训练A+B，seed10，从零50k；DF2K3450、HR256、有效batch64/micro16、L1、Adam2e-4 betas(.9,.999) eps1e-8、MultiStep[200k,400k,600k,800k] gamma.5、FP32、npy_mmap、eval/save1000。50k内不降LR。没有新loss、EMA、裁剪、分组LR或其它改动。复用A固定50k结果：最佳49k验证PSNR30.189747908588664。不能从A49k开始多训50k再与旧A比较；本版本不支持这种热启动归因。

A完整服务器备份：/home/featurize/work/span-x4-research/records/FEAT-LG6-SILU-50K-01-1792052706657616355，run子目录runs/FEAT-LG6-SILU-50K-01-LG6-SiLU-s10。参考A而非B0；server.py会核对A数据哈希、共同训练协议（含local_gate）及CUDA/PyTorch/GPU环境，记录A权重SHA256，不重训A/B0。

按最高DIV2K_valid PSNR选择一个权重报告全部测试集；补充各数据集峰值并标明权重。不得跨checkpoint平均PSNR。小幅单seed增益只作筛选证据，失败也完整保留。SiLU机制诊断/可学习beta不与本轮同时修改。

## 入口与产物

独立entry.py只替换模型factory，复用未修改的原trainer、sampler、metrics。prepare.py冻结单候选配置，server.py单候选队列默认不开自动释放，显式--auto-release才在训练成功且work备份校验成功后调用featurize instance release。训练/备份失败保留实例。先代码work独立checkout，datasets/runs/logs指向高速盘，再运行，不能在仍有进程的目录切换源码。

```bash
python3 research/gated_bypass/verify.py --device cuda
python3 research/gated_bypass/prepare.py --group FEAT-LG6-GB-50K-01 --data-root /home/featurize/datasets
python3 research/gated_bypass/entry.py --config logs/gated_bypass/FEAT-LG6-GB-50K-01/LG6-SiLU-GB.json --id FEAT-LG6-GB-50K-01-LG6-SiLU-GB-s10 --screen --screen-updates 50000
```

后台队列使用nohup python3 -u research/gated_bypass/server.py，参数--group、--data-root、--baseline-backup（A备份路径）、--backup-root（持久work）、--job，按需--auto-release。冻结config不能修改；恢复相同run加--resume。与其它队列共用本机logs/server-gpu.lock。

输出每步loss_steps.jsonl、每1000更新epochs.jsonl/逐图指标/编号权重、latest.pt、验证最佳best_validation.pt。backup备份完整训练产物和日志，SHA256清单/BACKUP_COMPLETE，curves/seed-10/LG6-SiLU-GB-overview.png包含loss/验证PSNR/验证SSIM三联图；comparison-50000.json的A为旧SiLU而非B0。原始数据与缓存不备份。

工程verify检查A初始化恒等、共有参数/RNG、奇数/矩形batch1/2、已知RGB相位修正与空间广播、首步R和后续G启动、有限梯度及20步更新、严格序列化续一步相同、部署一致和参数计数。integration检查真实入口累积、缓存、scheduler/RNG恢复与评测频率一致；engineering模式仅用于隔离合成fixture。本机检查不能等同于服务器已验证，也不能代替真实数据PSNR。

## CPU旁路诊断（2026-10-16）

diagnose_cpu.py只读取最佳验证权重，不运行主干、不训练、不释放实例。仅需100张原DIV2K验证LR×4 PNG（逐张匹配原manifest哈希），不需要HR或训练数据。输出门控参数、逐图空间变化、同RGB均值常色图响应差异、饱和比例，以及修正量RGB[0,1]/8bit等价RMS、相对输入幅值、替换门控为1或逐图均值对修正张量的影响；这些不是PSNR或独立重训消融。

排除1个LR像素边界，避免padding制造内容响应假象；均值/方差注明逐图等权与全空间加权，分位数为确定性采样近似。修正相对输入RGB，不冒称相对主干输出。没有真实LR图时只输出权重信息和NEEDS_REAL_LR_IMAGES，不用随机图代替证据。新报告保存在work，不改原权重/备份。

CPU合成已验证：恒定G=1、已知0.01修正RMS、输入相关空间响应、缺图保护；没有真实诊断结论。
