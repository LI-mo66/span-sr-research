# EASAResidual v1工程审查

日期2026-10-14；实现从d280dc5隔离分支新增，只修改research/easa目录。PyTorch2.9.1+cu128，本机RTX4060 Laptop。来源、形状与残差说明见README.zh.md。

## 已通过

- CPU/CUDA确定性检查：1×1、3×5、7×9、13×17、64×64、65×71，batch1/2，训练/推理输出均为N×3×4H×4W。
- 所有原SPAN参数和后续CPU/CUDA随机流与同seed baseline一致；诊断性gamma=0时，mid/tail两种接入均严格还原原SPAN输出。这只证明衔接，不是性能消融。
- 新分支不原地改写输入；不同图片不混合方差；零输入、非连续布局与1×1空间反向有限。
- 新参数全部进入optimizer且无重复；最终RGB L1反向有限；所有新增参数张量首步梯度非零、四步后均有真实更新。合成L1下降仅证明基本学习能力。
- serialize/strict reload无missing/unexpected；预合并Conv3XC后输出一致；新增8944、合并部署507152。
- CUDA确定性池化替代与CPU原生池化输出和梯度一致，含整除/非整除/并列最大值。

机器可读原始结果：engineering-cpu.json、engineering-cuda.json；每个参数的首步梯度和真实更新均保留。尤其beta等分支首步梯度很小（约1e-12量级），并非结构性零梯度，但尚不能据此判定真实数据上的学习效率。后续短筛应监测修正幅值与更新，不自动放大gamma或改变分组学习率。

## 失败与修复记录

首次CUDA在torch.use_deterministic_algorithms(True)下，报adaptive_max_pool2d_backward_cuda缺少确定性实现；不是PSNR失败。model.py加入完全相同floor/ceil池化区域和first-index max的确定性反向路径，标准LR64使用向量化网格。没有关闭确定性；随后CPU/CUDA完整检查通过。非整除大图确定性训练的逐区域路径可能慢，推理使用原生算子。

## 资源

engineering-resources.json记录本机FP32、LR180×320、5次warmup/20次同步测量、原路径和预合并路径、allocated/reserved及常驻模型范围。所有4个模型同时驻留，peak不是单个模型总显存。随机权重仅验证成本；本次预合并中位延迟baseline约11.99ms、EASA约15.58ms，说明小参数增量并不等于小延迟增量。不是4090/移动端速度结论。

## 尚未获得的证据

未进行真实DF2K训练或PSNR比较；尚未验证目标服务器、训练总预算收敛、跨seed稳定性、原官方预训练权重迁移、AMP。当前配置FP32；不同PyTorch环境须重跑CUDA验证。没有根据合成L1或论文分支消融宣称SPAN涨点。
