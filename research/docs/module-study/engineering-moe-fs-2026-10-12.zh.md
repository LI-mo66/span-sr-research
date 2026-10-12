# FusionGate4本地工程检查

2026-10-12，代码提交ddc0f92242d2a290a3f3af6f08e2dde6a838c24f；后续提交仅补充记录和文档。环境：E:/anaconda/envs/dl/python.exe，PyTorch2.9.1+cu128，NVIDIA RTX4060 Laptop。没有真实数据PSNR对照，也没有4090实机证据。

## 已通过

- `research/moe_fs/verify.py --device cpu`与`--device cuda`：SPAN52主干初始权重和CPU/CUDA随机流一致；零logits时train/eval输出与原模型逐元素一致；batch1/2、13×17与16×20空间形状；expert轴softmax；有限损失/梯度、门控参数3步实际更新；optimizer无遗漏/重复；严格state_dict重载；预合并部署输出一致。
- 实际构造计数：新增11024，部署509232，可学习2168160，总Parameter元素2632988。
- `resources.py`以LR13×17跑通CUDA资源报告：用于检查资源入口与计数，不作为论文速度或4090吞吐证据。输出`tmp/moe-fs-resource-smoke.json`。
- `check_queue.py`：两组除moe_fs开关外配置一致，拒绝改变冻结microbatch；固定末3个点、独立验证差值；模拟工作进程成功/失败/锁拒绝均保存正确退出码。
- Git Bash `-n`通过launch.sh与worker.sh语法检查；工作进程测试使用模拟Python/锁，不调用关机或服务器。
- `integration.py`：CUDA合成两次更新，effective_batch2/microbatch1，精确解码缓存；评测6个集合，写逐图指标、manifest、best/latest/周期checkpoint；损坏的未完成JSONL尾行恢复；从1步恢复到2步的所有模型state_dict张量与连续2步训练完全相同，scheduler一致。

合成运行ID：`ENGINEERING-MOEFS-fafb5cb427`与`ENGINEERING-MOEFS-fafb5cb427-continuous`。原始文件保留于本地`runs/`；fixture在`tmp/moe-fs-engineering-*`。合成随机HR/LR不是bicubic训练数据，其数值不能作为模型精度。

检查时发现`profile.py`遮蔽Python标准库导致Adam初始化失败，已在首个代码提交前改名为resources.py并重新完成CPU/CUDA检查。

## 保留限制

尚未在服务器验证实际路径、数据退化来源、版本、磁盘、显存和速度。服务器首次执行须先跑CUDA verify和数据preflight；预检确认配对/尺寸/hash不能证明bicubic退化来源。首轮仅B0/A1-G4各5k，之后由结果决定是否续到20k/100k。不保证无任何运行错误，不宣称PSNR增益。

原始SPAN源码与upstream c77a591比较无差异。本轮新代码位于research/moe_fs；共享训练器只在进程内替换模型工厂，源文件不修改。
