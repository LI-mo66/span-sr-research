# 独立 SPAN52 ×4 baseline

分支 baseline/span52-x4-seed10 从 788c90f 建立，不合并其他模块研究。
原始 SPAN52，seed=10，从零训练；DF2K 3450 对，HR256/LR64，
有效 batch64，micro16 累积4，L1/Adam，初始 LR5e-4，
MultiStep 200K/400K/600K/800K 减半，总计1M更新。
8线程PNG预取，FP32；自集成关闭；每10更新显示进度，
每5000更新评测独立 DIV2K_valid 和五个标准测试集。
原始论文未明确的优化器参数和重复训练阶段仍按 baseline_x4.json 记录；
这是明确冻结的论文设置复现协议，不承诺精确重现官方公开权重。

运行：激活环境后执行
`bash research/scripts/launch_featurize_full.sh SPAN52-X4-FULL-S10 /home/featurize/datasets`。
数据路径使用 downloads/Flickr2K 下 bicubic/X4，不使用 unknown 退化。
启动前做数据校验，nohup 后台运行；latest.pt 自动恢复，指纹不符拒绝。
配置写入 logs/server_jobs/实验ID，结果写入 runs/实验ID。
代码固定后不要在活跃训练目录更新源码。由 DIV2K_valid 选择最佳权重，
测试集各自 best 不拼成单个 checkpoint 的最终结果。

Featurize 自动停止计费接口尚未确认：此版本不调用 AutoDL shutdown，
不会自动归还实例。任务结束保存 status.txt，需平台控制台处理实例。
单一 seed=10 按用户要求固定；不作跨种子稳定性结论。
