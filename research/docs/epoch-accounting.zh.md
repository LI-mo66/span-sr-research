# epoch、更新次数与指标输出审计

论文第4.1节明确的是有效batch64、基础阶段100万次更新、每20万次减半LR；未披露dataset_enlarge_ratio或每epoch更新数。原始权重包只有四份test日志，没有train日志、train yaml或training state，因此不能从这些输出恢复作者真实训练epoch。

## 上游BasicSR代码

basicsr/train.py第45行的统计公式：

`日志估计的iters_per_epoch = ceil(N * dataset_enlarge_ratio / (batch_size_per_gpu * world_size))`

`日志估计的total_epochs = ceil(total_iters / iters_per_epoch)`

EnlargedSampler把数据重复后随机排列，epoch到下一轮就重设采样器。注意训练DataLoader设置drop_last=True，非整除情况下实际循环batch数可能比ceil日志估计少一个；精确口径还需读取len(train_loader)，不能只信启动打印值。多卡每rank样本数也经过ceil。作者SPAN的ratio未知。

仓库EDSR_Mx4示例使用800个DIV2K样本、ratio100、单GPU batch16，因此每epoch5000次更新；total_iter300000对应60个完整epoch。它每100次更新输出当前batch loss，每5000次更新评测Set5 PSNR，并非每epoch自动计算训练loss均值，更没有该配置下五个标准集的SSIM。此例只能说明代码运行方式，不能当SPAN正式配方。

如果假设SPAN训练用DF2K3450、有效batch64、ratio100，则启动统计ceil(345000/64)=5391次；但实际drop_last循环需核对。不能将未披露的ratio100代入后宣称作者采用该值。

## 当前独立研究入口

ratio未设置；每research epoch明确执行ceil(3450/64)=54次优化器更新，使用随机有放回采样，非上游EnlargedSampler实现。每次更新16个microbatch4，合计有效batch64。一个epoch共抽取3456对随机裁块，可能包含重复图片。

训练日志补齐：每epoch打印并在epochs.jsonl保存updates_in_epoch、累计update、samples_in_epoch、train_l1_mean和train_l1_last。train_l1_mean是本epoch所有有效batch的L1均值；固定batch时每次更新等权。L1是在RGB浮点输出/HR上计算。每个epoch结束时，独立validation与五个标准集做完整评测，分别逐图计算Y PSNR/SSIM再取算术平均，不把训练patch PSNR当测试集PSNR。

epoch仅作为记录单位，LR/停止预算始终按实际optimizer update计数。若日后改成每5000更新一次记录，100万更新就是200个记录周期；这只是项目约定，不是发现了作者的epoch。不能让改动记录周期同时改总更新数/衰减计划/采样器，然后误称条件一致。

基础阶段54次/epoch对应18519个research epoch、末轮28次。20轮为1080更新，过短而不足以判断百万更新下的最终性能。正式筛查应在baseline学习曲线基础上预先固定更新预算，再把epoch定义写清。

公开权重五集评测没有训练迭代，也没有训练epoch或训练loss；仅有该checkpoint的PSNR/SSIM。重新评测不能恢复它过去的loss曲线。
