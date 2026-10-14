# SPAN52-EASAResidual v1

独立分支 `research/easa-mid`，从 `d280dc5` 建立；原始SPAN、MoE-FS实现、既有队列和其他分支不修改。本目录新增一个模型工厂、训练入口、配对配置、工程验证与资源测试。工程PASS不等于PSNR上涨。

## 结构与激活

推荐E1：原SPAB3输出 → EASA残差适配器 → 原SPAB4。原六个SPAB、Conv3XC和四路融合顺序保留。block6的第一路返回值按原实际forward使用，不当作未激活卷积重新构造。可配置tail位置仅供以后独立位置对照，默认mid；不叠MoE-FS/其它模块。

公式：`Z=L2Norm_C(F)`；`X=Conv1(Z)`；`S=DWConv3(AdaptiveMaxPool(X))`；`V=Var_HW(X,correction=0)`；`M=ResizeNearest(GELU(Conv1(alpha*S+beta*V)))`；`Delta=Conv1(X*M)`；`F'=F+gamma*Delta`。

所有投影52→52且有bias；depthwise3×3、padding1；缩放因子8；alpha1、beta0、每通道gamma0.01。投影采用PyTorch默认非零随机初始化。gamma小但非零，不把修正路径同时全部置零。原主干先初始化，新增模块使用隔离随机流seed+20001，不消耗后续主CPU/CUDA随机流。

只在调制生成内部使用GELU；残差相加后不增加ReLU/额外sigmoid/clamp/归一化，保留有符号信息。L2归一化只作用于新分支，不改变原残差。不是原完整SMFA/FMB，也没有LDE、PCFN或显式远距离像素匹配。

## 接口与尺寸

|节点|尺寸/约束|
|---|---|
|输入RGB|N×3×H×W，沿用原mean/img_range，×4|
|EASA输入F、输出F'|N×52×H×W，NCHW、同dtype/device、无原地改写|
|粗网格S|N×52×max(1,H//8)×max(1,W//8)|
|方差V|N×52×1×1，只沿空间轴，图片之间不混统计|
|恢复后的M、Delta|N×52×H×W，nearest指定原H/W，无窗口补齐/额外crop|
|四路融合|原顺序concat→N×208×H×W→原conv_cat→N×52×H×W|
|最终输出|N×3×4H×4W|

新增参数alpha/beta/gamma和卷积均注册入optimizer；无跨图缓存、EMA或更新buffer。M无[0,1]限制，特征和修正可有正负值；只有既有评测环节量化。保存使用严格state_dict；旧SPAN权重与EASA完整权重不同，禁止把strict=False当完整加载证据。

## 确定性池化兼容

首次CUDA检查发现PyTorch2.9.1的adaptive_max_pool2d_backward_cuda不支持确定性算法，原生路径失败记录保留在工程说明。没有关闭生产训练的确定性设置。

新增adaptive_structure_pool：CUDA反向且要求确定性时，整除网格通过reshape/permute/max实现相同池化区域；非整除网格按原adaptive pooling的floor/ceil边界逐区域取first-index max。输出与并列最大值梯度对照原生CPU实现；推理沿用原生池化。非整除大图的确定性训练可能较慢；默认LR64训练走向量化8×8网格。方差用总体方差确保1×1空间有限。

## 参数与训练设置

新增8944参数，SPAN52合并部署498208→507152（+1.795%）。LR64单张新增卷积22,354,176 MAC，不包含池化、归一化、方差、激活、resize与乘法，不能称全算子FLOPs。

本分支沿用最新已有的 `multistep-lr2e4-v1` 配对协议：baseline与E1均LR2e-4、总1M更新、200k/400k/600k/800k减半、RGB L1、HR256、effective batch64、microbatch4、FP32；每1000更新评测，Y通道/crop4/无自集成。它不同于最初讨论的LR5e-4协议，因此不得与旧5e-4结果直接归因比较。低学习率只是已有研究协议，未证明修复全部稳定性问题。

训练入口重用当前 `research/moe_fs/trainer.py` 中的日志、cache、checkpoint和恢复功能，仅替换本进程的build_model；这个文件是共享训练器，此处不会启用FusionGate4。真实数据入口检查DF2K3450、独立验证100及五个benchmark数量、全部图像可解码和×4尺寸、hash重叠。需要单独审计bicubic来源与环境；预检不代替这些证据。

## 使用（仓库根目录）

```bash
python research/easa/verify.py --device cpu
python research/easa/verify.py --device cuda
python research/easa/integration.py
python research/easa/resources.py --output results/easa-resource.json
```

先按 `research/configs/paths.example.json` 创建本地paths文件，再分别预检。下面命令需要CUDA与真实数据，只是操作说明，本次未启动真实训练：

```bash
python research/easa/run.py --config research/easa/configs/B0-lr2e4.json --id B0-EASA-s10-unique --preflight
python research/easa/run.py --config research/easa/configs/E1-mid-lr2e4.json --id E1-EASA-s10-unique --preflight
python research/easa/run.py --config research/easa/configs/B0-lr2e4.json --id B0-EASA-s10-unique --screen --screen-updates 5000
python research/easa/run.py --config research/easa/configs/E1-mid-lr2e4.json --id E1-EASA-s10-unique --screen --screen-updates 5000
```

两组同数据、seed、预算和环境；训练源码须已commit且clean。工程集成只生成合成图像，2次更新验证cache/累积/保存/恢复，不称真实PSNR。独立验证选checkpoint，公开测试只监测；短筛不得作为最终涨点结论。

## 来源与适配差异

ECCV2024 SMFANet正式论文：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf

原官方代码：https://github.com/Zheng-MJ/SMFANet/blob/main/basicsr/archs/SMFANet_arch.py

本代码为独立适配，借鉴EASA粗尺度/方差调制机制。论文公式直接加方差，官方代码使用alpha/beta且beta初始0；本方案遵循代码的系数初始化，但明确选择correction=0而非torch.var默认修正。新增输入/输出投影、外层小残差scale、小尺寸保护与确定性池化；不宣称完整SMFANet复现或未经验证的收益。
