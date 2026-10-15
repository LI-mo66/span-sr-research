# 实验索引

2026-10-12：已建立记录入口；本轮没有启动真实数据训练，没有可填写的PSNR数据。

|run ID|方案版本|匹配baseline ID|seed|更新数|独立验证PSNR/差值|公共测试结果|部署参数/延迟/显存|状态与记录路径|
|---|---|---|---|---|---|---|---|---|

运行开始后追加一行；实验中断/失败也必须保留。逐图指标、日志和checkpoint放`runs/<run-id>/`，本表指向`records/<run-id>.md`。不同协议不直接填相对提升。

工程计数记录（不属于PSNR实验）：2026-10-12，当前FusionGate4新增11024参数，详见moe-fs-plan.zh.md；准确代码commit/dirty与环境在真正实验冻结时另行记录。

独立实现工程验证：代码ddc0f92，CPU/CUDA与合成两步恢复检查通过；见engineering-moe-fs-2026-10-12.zh.md。合成运行不填入真实数据PSNR主表。


2026-10-15：新分支 research/spab-linear-gate6，LG6-SiLU/LG6-ReLU 设计及独立实现完成。本机CPU/CUDA合成、入口累积/缓存/断点恢复检查通过；没有真实PSNR结果。见 research/linear_gate/README.zh.md 与 engineering-2026-10-15.json。A1-G4保持停止，复用基线指向 results/FEAT-MOEFS-G4-LR2E4-20K-01；本轮实验结束后提醒讨论方案2图像旁路。
