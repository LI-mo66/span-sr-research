# A6交付工程证据

2026-10-15，本机RTX4060 Laptop、PyTorch2.9.1+cu128；目标服务器4090/PyTorch2.2.2+cu121尚未运行，launch会在目标环境重查。

|检查|结果|证据|
|---|---|---|
|CPU与确定性CUDA，采样坐标oracle及输入梯度|PASS|engineering-cpu.json / engineering-cuda.json|
|主干初值与CPU/CUDA RNG、六处独立参数、gamma旁路等价|PASS|同上|
|完整L1 loss有限梯度；六处所有新参数四步内实际改变|PASS|同上；不将小梯度数目解释成PSNR收益|
|1×1、3×5、13×17、64×64、65×71，train/eval与不连续特征|PASS|同上|
|strict state_dict保存加载、预合并输出对齐|PASS|同上|
|实际micro16/累积4随机合成训练内存|PASS|engineering-resources.json，allocated3072456192字节，reserved3405774848字节|
|协议拒绝、baseline训练拒绝、路径恢复、验证选择与fixed/last3、逐图对照、完整曲线输出|PASS，3个测试|check_contracts.py；其中示例PSNR为合成fixture，不是实测精度|
|未改shared trainer的真实入口、npy cache、梯度累积、checkpoint和日志损坏尾行恢复|PASS|runs/ENGINEERING-A6-69ce71671d（本机忽略目录，不打包）|
|分段1+1更新 vs 连续2更新|模型所有state_dict元素、scheduler、逐步loss/LR完全一致|同上及-continuous；测试源码020f696|
|三个服务器Bash入口语法|PASS|Git Bash -n逐文件检查；没有执行真实服务器命令|
|共同源码与d280dc5差异|空|basicsr / research/moe_fs/trainer.py / research/runtime.py / research/image_io.py / research/parallel_loader.py|

资源秒数是合成随机输入，不含真实数据/完整验证开销，不能作为50k训练ETA。存储参数包含Conv3XC冻结eval核，部署参数为一次预合并模型口径。

没有启动真实数据训练，没有训练B0，也没有远程提交训练任务。没有PSNR涨分结论。只交付服务器代码与可检查的执行入口。
