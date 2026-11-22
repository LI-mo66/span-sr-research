# 实际训练与评测协议

## 训练

原始初始化从零开始，seed10。DF2K共有3450对，其中DIV2K训练800对、Flickr2K 2650对。Flickr2K实际位于 `/home/featurize/datasets/downloads/Flickr2K`。LR裁剪64×64，HR裁剪128×128，RGB输入输出。

RGB L1损失，Adam的betas为(0.9,0.999)，eps为1e-8。有效batch64，micro-batch16，梯度累积4次。FP32。每1000次优化器更新定义一个research epoch并进行评测和保存。

初始学习率2e-4，MultiStepLR在累计200K、400K、600K、800K更新处执行gamma0.5。里程碑当次Adam更新使用之前的学习率，之后scheduler.step将下一次更新的学习率减半。loss_steps.jsonl记录scheduler.step之后的学习率，不能把该条记录直接当成这次Adam已使用的学习率。

| 实际Adam更新区间 | 学习率 |
|---|---:|
| 1–200000 | 2e-4 |
| 200001–400000 | 1e-4 |
| 400001–600000 | 5e-5 |
| 600001–800000 | 2.5e-5 |
| 800001–1000000 | 1.25e-5 |

0–199K阶段未设置梯度裁剪。在原轨迹199458更新触发损失保护后，正式保留的轨迹从199K完整断点恢复，199001起启用全局梯度L2范数裁剪0.1。裁剪在四个micro-batch反向累积完成后、Adam.step之前执行。模型、Adam、scheduler及Python/NumPy/Torch/CUDA随机状态随阶段完整恢复。×2后续没有提前减半学习率。

有效配置的初始manifest仍保留total_updates=50000及原指纹。服务器阶段包装扩展执行上限并加入裁剪，后续审计必须结合阶段脚本与provenance，不能仅依赖初始manifest判断训练设置。

## 环境

用户验收环境为RTX4090、PyTorch2.2.2+cu121、CUDA12.1。deterministic=True，cuDNN TF32=True，matmul TF32=False。验收报告的配置协议指纹为 `1cd930967139a355d9adea5876e71a1bdb8deedd57fd06eb34a632a5cd22b1ae`。

## 评测

DIV2K_valid100、Set5 5、Set14 14、B100 100、Urban100 100、Manga109 109，共428对。Y通道PSNR/SSIM，裁边2像素，整图推理。输出裁到[0,1]并量化为uint8后计算指标。普通与D4八变换自集成均记录，自集成用于推理评测。按普通DIV2K_valid最高PSNR选一个权重，最早并列优先，所有集合和×8统一使用该权重。

## 后段检查

600K→800K裁剪2730/200000次，最大整步L1为0.0206064。800K→1M裁剪1458/200000次，最大整步L1为0.0211808。全程1000000条逐步损失记录连续、有限，1000次评测完整。

末两个10K窗口的普通DIV2K_valid均值为36.275030和36.274677，表明后期增益已较小。数据支持当前轨迹后段稳定，长期或其他倍率的稳定性仍由各自试验判断。
