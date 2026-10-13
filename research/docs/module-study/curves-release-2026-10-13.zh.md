# 绘图、持久备份与Featurize释放

用户确认未启动短筛，授权运行完自动关机。服务器help确认featurize instance release [instance_id]，缺省为当前实例。实际是释放实例，非仅停止Python；高速本地盘可能被销毁。

独立trainer.py复制当前研究训练器，只增加每个optimizer更新的RGB L1及LR记录loss_steps.jsonl；恢复时按checkpoint更新数截断未提交/损坏尾行。没有改变模型、loss、采样、优化器或学习率算法，原SPAN及research/train.py不修改。

curves.py：每run的原始每步训练loss、周期均值loss、6个评测集PSNR/SSIM；两组loss叠加及每集PSNR/SSIM叠加和同更新数差值。完整两组正常结果23张图，每张PNG/SVG，保存源数值source-data.json。不是给每个测试集编造训练loss，没有插补缺失评测点。

AUTO_RELEASE=1启用：持有GPU队列锁的worker退出（正常/训练错误）后，先保存status，再复制组logs和所有已有seed的run全部原始文件/checkpoint到BACKUP_ROOT，每个源文件/复制件比对SHA256，生成曲线及backup-manifest.json/BACKUP_COMPLETE，sync后调用平台release。训练失败没有可用曲线时记录原因，仍备份已有文件。正常训练绘图失败、任何备份失败均拒绝释放；锁拒绝不释放。实际平台API失败需人工停止，SIGKILL/崩溃不保证能执行退出处理。

本地通过：23组PNG/SVG绘图与哈希备份检查；模拟成功、训练失败、备份失败、锁拒绝的释放顺序（无真实释放）；Linux脚本语法；CUDA合成2更新的恢复/连续及评测节奏一致性，loss_steps损坏/超checkpoint尾行恢复。运行ENGINEERING-MOEFS-1bb0b90c03。未真实调用服务器release，不保证平台API可达。

BACKUP_ROOT推荐/home/featurize/work/span-x4-research/records。数据集和解码缓存不备份到work以免慢IO与配额问题；释放后若续训必须重新准备相同数据、环境、源码并恢复checkpoint。
