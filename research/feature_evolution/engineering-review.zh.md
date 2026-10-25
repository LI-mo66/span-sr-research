# SG6-B3特征演化：工程审查

日期：2026-10-25。结论：加法/差分候选的合成接线与学习启动检查通过，可作为后续真实数据短筛的工程起点；没有真实PSNR、真实特征分布匹配或长期稳定性结论。

## 改动及保护

只在新目录增加候选工厂、入口、配置和验证工具，继承固定SG6构造器；未修改原SG6/Conv3XC、训练器、loss、optimizer、scheduler、数据或指标代码。
内部为a=SiLU(C1(x))、b=SiLU(C2(a)+s*a)、c=C3(b)，F=c+Pout(SiLU(concat(Pi(stage))))。
外层仍为(x+F)*[local_gate(F)+spatial_gate(F)]；F是送入门控的特征，融合支路输出是修正量，没有再加一遍输入。
Pi正常初始化，只有Pout权重和偏置置零；此初始化只消除新增融合修正，不使候选初始函数等于SG6。

实现与两组真实入口合成测试的本地源码commit：01a0b4a2f096b5365b73e0af7a051256191b0e61（父提交固定SG6 e731483）。随后追加本报告、推理剖析及日志，不改候选/训练入口。GitHub发布提交会另含本报告，commit不同；不能拿本地工程commit冒称服务器训练来源。
本地是独立checkout，对象借用原仓库只读Git对象；正式服务器应从GitHub分支独立clone并冻结最终commit。

## 用户关切逐项处理

|问题|实现与证据|边界|
|---|---|---|
|尺寸衔接|输入/输出NCHW、52通道；各阶段投影13通道，拼接39，再投回52；显式断言三个阶段shape一致|不通过广播/reshape掩盖错位|
|空间对应|投影全为1x1，原3x3 padding/stride保持；无缩放、窗口、重排、mask或额外裁剪|合成测试覆盖1x1、13x17、16x20、17x13，batch1/2，输出严格4倍|
|dtype/device|随主干迁移；CPU/CUDA FP32及FP64等函数诊断通过|未测AMP/BF16；原协议FP32，不自动启用AMP|
|特征分布|融合修正从零启动；保留c直通，未强加Sigmoid或归一化；记录12步后delta与两个门控RMS/absmax|真实图像上的阶段尺度与幅值匹配尚未证实；投影只衔接通道，不被当作分布匹配证明|
|优化逻辑|仍同一L1和Adam2e-4；所有trainable参数梯度有限，optimizer无遗漏/重复；每个新增参数实际更新|只能排除测试范围内断梯度/死分支，不能证明无真实任务梯度竞争|
|启动|Pout首步有非零梯度；Pi首步梯度为零符合预期，后续12步内每个Pi权重/偏置均更新|没有同时将整个乘积路径置零；无附加学习率组、warmup或裁剪|
|归一化堆叠|完整模型检查无BN/LN/GN/IN，新增归一化为0|新增融合SiLU非原地，不据幅值猜测预防性加norm|
|残差与原地操作|a保留原返回的激活后out1语义；b的SiLU作用在新张量上，不覆写a；c未原地修改|没有把外层残差加法改成减法；B3之外保持原工厂|
|状态与恢复|严格state载入无missing/unexpected；模型及Adam状态重载后下一步更新逐bit一致|sign由配置冻结，单独state_dict不能验证sign；正式续训由原训练器config/source/data/environment fingerprint拒绝错配|
|部署|CPU与受控CUDA精度下train/eval数值比较、Conv3XC融合推理一致|默认TF32下的数值敏感性单列；不声称默认精度逐bit一致|

共享原层与新增分支RNG隔离已检查；加法与差分版原始state相同，但不等于初始函数相同。
FP64实测：把差分版C2的可学习捷径权重加2I，就能匹配加法版全部块返回值（rtol/atol 1e-10）。这是固定减号不必然扩大函数类的反证，保留在正式讨论中。

## 工程证据

- engineering-cpu.json：两组、每组12步L1合成更新，尺寸、有限性、分阶段启动、实际更新、严格重载、部署检查PASS。
- engineering-cuda.json：同上；RTX4060 Laptop、torch2.9.1+cu128。验证进程关闭TF32并启用确定性计算，正式训练器精度策略未变。
- integration-difference.txt：ENGINEERING-FE-f8503eddc6，两步合成数据，梯度累积/mmap缓存/损坏尾日志恢复/优化器和调度器恢复/连续与续训模型逐bit比较PASS。
- integration-add.txt：ENGINEERING-FE-6cd0ffb07a，同范围PASS。
- 合成数据是随机生成的HR/LR，日志中的PSNR完全不作为超分性能证据；没有使用或训练真实数据。
- initial-failures.json、precision-probe.json保留初次失败与受控定位；失败未删除。

首次CUDA训练/融合比较absmax约9.47e-4未过原阈值。受控13x17探针中，原SG6/加法/差分在TF32下均约2.71e-3，关闭TF32后约1.19e-5至1.24e-5，支持该合成条件下TF32是主要数值差异来源；不外推到所有真实权重与设备。
首次续训逐bit检查未启用确定性算法，随后按原训练器的确定性策略检查通过；未放宽逐bit要求。
初始来源是源码archive，没有Git元数据，真实入口被clean-tree检查阻断；补建隔离Git checkout并提交候选后才运行，未绕过生产保护。

复查命令（在仓库根目录）：

```bash
python research/feature_evolution/verify.py --device cpu --output /tmp/fe-cpu.json
python research/feature_evolution/verify.py --device cuda --output /tmp/fe-cuda.json
python research/feature_evolution/integration.py --variant SG6-FE-add
python research/feature_evolution/integration.py --variant SG6-FE-difference
```

后两项仅运行合成数据工程检查，会在tmp/runs生成新fixture，要求Git源码干净及CUDA。它们不构成50K训练启动命令。

## 成本

部署计数：SG6 535772；两候选均539919，新增4147（约0.77%）。
新增卷积MAC为4056*H*W；不含SiLU、concat、加减与原模型，不宣称完整FLOP覆盖。
profile-cuda.json：本机RTX4060 Laptop，float32、TF32关闭，batch1/LR64x64，融合后eval，5次warmup、20次逐次同步计时。

|网络|推理中位ms|peak allocated字节|peak reserved字节|
|---|---:|---:|---:|
|SG6|15.5946|15788032|39845888|
|加法|16.1730|15807488|39845888|
|差分|16.13445|15807488|39845888|

该次加法约+3.71%、差分约+3.46%；顺序测量、未训练权重、小尺寸、单机结果，不能当4090服务器吞吐保证。reserved包含分配器缓存，不是有效特征内存。
训练吞吐、训练峰值显存、真实尺寸延迟与跨图缓存驻留未测；本模块没有跨图特征缓存。

## 下一步与停止边界

工程上没有发现尚未解决的shape/像素错位、永久死路径或新增norm堆叠；不能因此宣布真实分布、任务优化或PSNR问题已解决。
真实短筛需核对共同协议与数据，固定验证ROI，并观察阶段特征/delta/门控尺度、有限更新、损失与逐图PSNR。出现异常先定位，不默认叠加norm、剪裁或改loss。
保持原方案的SG6/加法/差分对照及停止线；没有自动启动真实训练、服务器任务或关机动作。
