# 标准 SPAN ×4：训练方法与公开权重测试

范围：官方 SPAN，48 通道、6 个 SPAB、×4；不是 SPAN-F、52 通道版本或 28 通道竞赛模型。来源：用户提供的 CVPRW2024 原论文第4.1节；锁定的官方源码；公开权重包内 SPANX4_CH48 测试日志。

## 论文明确披露的训练方案

| 项目 | 设置 |
|---|---|
| 训练集 | DF2K = DIV2K800 + Flickr2K2650，共3450张HR |
| 退化 | HR经bicubic下采样构造LR；×4独立训练 |
| 网络 | RGB输入/输出；特征48通道；6个SPAB；训练Conv3XC多分支，推理合并卷积 |
| 裁块 | 每个样本HR256×256；对应LR64×64，空间坐标成对对齐 |
| Batch | 全局有效batch64个图像对 |
| 增强 | 随机翻转、旋转；LR和HR共享变换 |
| 损失 | Adam优化L1像素损失；在RGB图像上训练 |
| 初始LR | 0.0005 |
| LR衰减 | 每200000次优化器更新减半 |
| 基础训练长度 | 1000000次更新 |
| 作者硬件 | 4张NVIDIA A100；论文报告一次训练约12小时；不能移用作4060耗时承诺 |

对应一个基础训练阶段的LR时间轴：0–200k：5e-4；200k–400k：2.5e-4；400k–600k：1.25e-4；600k–800k：6.25e-5；800k–1M：3.125e-5。阶段边界的scheduler调用顺序必须固定，不能让对照与改进相差一个更新。

## 每次更新如何执行

1. 从DF2K取64个HR/LR图像对，各自随机选择对齐的HR256/LR64裁块。
2. 对HR与LR施加相同的随机几何增强，避免像素位置错配。
3. 送入SPAN得到RGB超分输出；输出不预先clamp或量化。
4. 计算 `mean(abs(SR - HR))`，反向传播，Adam更新一次，然后按固定更新计数推进学习率计划。
5. 保存训练状态；在固定评测点做eval并统计Y通道指标。我们的工程额外要求每研究epoch评测全部五个标准集并记录各自best，但仅DIV2K_valid选择方案。

这是结合论文与代码给出的可执行解释；论文并没有披露每一步采样器实现、增强概率或scheduler调用顺序。

原始forward将[0,1] RGB减均值[0.4488,0.4371,0.4040]后乘255，再进入网络；输出直接返回。加载官方params后，Set5首图原始输出约[-0.00345,1.00349]，直接作为[0,1]图像评测正常。不要额外把输出除255或再次加均值。训练目标为未量化的[0,1] HR；指标计算时再clamp、乘255并round为uint8。

## 重复训练阶段：尚不能完全还原

论文还描述加载已训练权重后重复上述训练设置两次。不能简单把已发布模型理解为只训练1M更新即可重现。论文没有完整给出总阶段解释、Adam动量是否重置、scheduler如何重启、以及此公开文件对应哪个阶段。当前baseline_x4配置只实现第一阶段，不能宣称完整论文表格训练复现。

## 未公开或不能由测试日志证明的细节

- Adam的betas/epsilon、weight decay；当前工程选择(0.9,0.999)、1e-8、0，这些是显式默认假设。
- 精确随机种子、分布式采样顺序、增强概率、AMP、梯度裁剪、warmup、EMA训练decay。
- checkpoint同时含params与params_ema，说明存在两套保存参数；不能仅凭它反推出EMA训练方法。包内官方测试日志明确使用params。
- 测试日志还列出SRFormer teacher和SRModelRAW，但完整对应训练实现未随这个包提供。出现teacher配置不等于证明标准论文训练使用蒸馏；也不能武断断言这个公开checkpoint完全没有蒸馏。
- 上游options/train只有EDSR配置，没有可直接引用为官方完整SPAN训练配方的配置。网络名在包内为AttLiP18、在公开源码中为SPAN；204个state键严格匹配，但键匹配本身不是训练来历证明。

## 本机研究协议与作者设置的区别

本机单张4060 Laptop GPU 8GB，PyTorch2.9.1+cu128。当前工程采用FP32，microbatch4、累计16次反向传播后执行一次Adam更新，使有效batch64。每个microbatch平均L1再除16，可得到等大小batch下的总平均梯度；浮点累加顺序仍与四卡全batch不同。没有启用AMP、teacher、EMA或自集成；这属于本项目从零训练控制协议，不能称为所有作者细节均已复现。

当前研究epoch定义为ceil(3450/64)=54次更新，随机有放回采样；20个epoch只有1080次更新，并非官方100万次更新的等价缩短。它只能检查启动和早期趋势，无法证明最终稳定收益；正式短筛预算需要结合baseline学习曲线预注册。BasicSR的dataset_enlarge_ratio还会改变epoch含义，比较时必须比较实际更新次数。

## 公开权重测试口径与命令

params；RGB[0,1]输入；原始官方eval；全图batch1；FP32；不裁块、不自集成；输出clamp/round到uint8；Y通道PSNR/SSIM；裁4个HR边缘像素；逐图计算再取均值。保留权重/data/source哈希、逐图指标与日志。

在PowerShell中：

```powershell
Set-Location E:\SPANimagebase\span-research
E:\anaconda\envs\dl\python.exe research/evaluate_weights.py --id YOUR-UNIQUE-ID
```

默认权重weights/official/spanx4_ch48.pth、默认key=params；自动测试五个标准集。每个数据集保存第一张SR图；加`--save-images`保存全部SR图。每次用新ID，已有结果不会覆盖。输出位于results/<ID>/，包含summary.json、console.log、manifest.json及五份逐图指标JSON。

公开权重测试属于外部checkpoint评测，可以核对评测流程；不能替代本项目固定数据和更新预算的从头训练baseline。

来源链接：https://arxiv.org/html/2311.12770v2 、https://github.com/hongyuanyu/SPAN 、weights/official/SPANX4_CH48/test_AttLi_x4_20231123_155330.log。
