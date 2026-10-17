# GB 六块空间门控修正 SG6

候选主问题：在保持 GB 简单图像旁路的条件下，额外利用已有特征的邻域来生成门控，能否超过已归档 GB？不声称必然提高 PSNR。

每块保持原有三层主卷积、SiLU、线性门控、残差相加位置和返回的 out1 语义：

`F=c3(SiLU(c2(SiLU(c1(X)))))`

`M=local_gate(F)+project(SiLU(depth3x3(F)))`

`Y=(X+F)*M`

六块结构相同、参数独立。所有特征为 NCHW、B×52×H×W；depth groups=52、padding=1、stride=1，投影为52→52，无空间错位或广播。新 SiLU 非原地，不修改 F。负门控保留，不加 Sigmoid/Softmax/归一化。GB 的图像旁路仍用 RGB [0,1]，主干预处理仍为 (image-mean)*255；两者的既有融合和 PixelShuffle 均保持。

构造完整 GB 后，在隔离 RNG 内新增分支。depth 正常初始化，project 权重/偏置置零。起点是同初始化 GB 的精确映射，不是已训练最佳 GB 的映射。首步 depth 梯度为零，project 启动后 depth 应开始学习；verify 检查六块全部启动。生产训练从零50K，不加载 GB 权重。

参数：depth 每块520，project每块2756，合计3276；六块19656。GB部署516116，SG6部署535772，相对B0多37564。新增卷积MAC为六块×(52×9+52×52)×H×W=19032HW，未含激活、门控加法等开销；实际延迟未知。六块并不共享参数，分支不保证自动归零或稳定。

保持seed10、Adam2e-4、HR256、有效batch64、micro16、L1、原MultiStep完整1M时间轴，训练截停50K，每1K评测六集合。固定验证集最高PSNR选权重、同权重报告所有集合，另列各集合峰值和权重；无时间平均。GB/PB旧结果复用；新队列不会调度它们。启动审计GB共同协议（含原image_bypass）、所有图像哈希和环境。

MS负例：复杂化图像旁路未超GB/PB，本方案不再改该旁路。门控推理诊断是固定权重敏感性，不能证明新增邻域能力必然有效。50K无优势就停止本版本，不用额外预算或叠模块解释失败。

## 服务器

独立clone本分支后：

```bash
bash research/spatial_gate6/launch.sh FEAT-LG6-SG6-50K-01 /home/featurize/datasets /home/featurize/work/span-x4-research/records/FEAT-LG6-GB-50K-01-1792082028817331043
```

nohup后台训练，无关机/释放命令。自动绘图及SHA校验备份到work/records。必须检查training_exit_code和backup_exit_code；仅有BACKUP_COMPLETE不代表训练成功。失败保留现场。同组重启只在源码/配置/数据/环境指纹一致时续训；不要对运行中的代码checkout。

输出包括latest、best_validation、每1K断点、loss_steps.jsonl、epochs.jsonl、逐图指标、冻结配置、comparison-50000.json、Loss/PSNR/SSIM三联图、SHA备份清单。训练实时文件在高速盘，完成后备份work；实例保留。

工程验证范围：CPU/CUDA合成训练、同起点GB严格一致、非零GB参数映射与返回值、长方形/批量1/2、梯度分阶段启动、严格保存恢复、部署融合；另用真实训练入口验证两步合成数据的续训与连续训练一致性。测试结果见engineering.json，合成测试不代表真实PSNR或长期稳定性。
