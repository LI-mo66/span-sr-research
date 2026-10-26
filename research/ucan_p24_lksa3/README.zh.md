# SG6：LKSA顺序、布局与Q/K复用三项短筛

本分支直接从固定SG6提交e731483d4ce4e325f3d64dbbc80add342b46f7f1建立，只增加自身实验目录，不改原SPAN、SG6、旧实验源码及产物。用户固定只测seed10；从零50K，原LR2e-4、Adam/MultiStep、HR256、micro16/effective64、每1K评测、不自集成。按DIV2K_valid最高PSNR选择一个权重报告六集合，另报固定50K与40–50K尾段；不据单seed宣称普遍收益。

三项：P24+LKSA3、P26+LKSA5、P24复用Q/K+LKSA5。直接对照已成功的P24+LKSA5（d42b73fef5b2ec3e088f340ab033a5c4ac13a190、48K DIV30.28196911823231），并保留SG6和原P24比较。每项独立分支、目录、配置、ID、备份；不热启动或自动补seed。

完整LKSA、RMS输入输出适配、零投影、上下文attention.py均沿用旧实验。LKSA3只是把同一单元移至第3块后、context4之前；P26只移动第二个上下文至第6块后。各自公共参数初值与旧组合逐张量一致、骨干初始化与训练RNG一致，LKSA始终seed+90001独立初始化。前两项参数565087，复用版562331。位置比较包括原融合头的既有路径差异：第6块first中间特征仍不被其输出后的适配器改写。

复用版先按原seed+80001顺序创建两个独立单元，再删除receiver q/k，标记reuse=True。因此其余参数不因删除Q/K而改变初始化。producer返回原始Q/K，receiver使用当前特征的branch-pre-LN重新生成V，保留自己的Hedgehog映射、Fourier系数、温度、LePE、投影和RMS写回。不重复复用完整输出或已计算的注意力矩阵；Q/K不detach，只作为当前forward局部变量。H/W、batch、dtype、device必须匹配，不能仅靠token数相等。C52/half26/head1合法，无新归一化层或额外残差。

复用版receiver保留原52维pre-LN，统计量为当前V服务；producer Q/K和receiver V不保证训练后分布自然匹配。这里将其作为待验证假设，而非通过额外归一化/门控预先修正。原receiver映射、温度和投影可学习适配；先过有限值、梯度、真实更新、批次隔离、源公式、端到端测试，不通过则停止，不静默改变优化器。

验证包含：旧组合公共初值逐张量比较；独立和复用源SDFL/DFRL在source_signed模式前后向等价；生产abs_sum模式公式与异常状态检查；LKSA源公式等价；非零修正下原forward+hook oracle；奇数长宽/批次/FP64部署等价；全部新增参数真实更新；严格model/Adam/MultiStep保存续训；receiver-only loss回传producer Q/K，同时验证V由当前特征生成；四步真实训练入口与缓存/累积/部分日志恢复。source_signed只做源等价，不用于实际训练。

预检在同一进程、相同随机权重初始化/尺寸/精度/同步计时方式下，分别测SG6、原P24、旧P24+LKSA5和当前候选；保存原始时间样本、完整路径峰值显存及生产形状合成训练显存。没有用旧批次10ms与新批次7ms直接推断加模块会加速。profiling不加载训练权重，不作为最终部署延迟证明。组件单独计时保留原始QK状态，不计为完整推理时间的可加分解。

先通过三项全部服务器预检，再nohup串行从零训练。全程共用GPU队列锁，不终止别人的进程；GPU忙或锁被占用则拒绝启动。每项成功后独立备份，失败也保存可获得日志；任一步失败停止队列并保留实例。三项全部成功、三个持久归档再次hash校验、汇总ZIP校验且GPU无其它计算进程时，才按用户既有授权申请释放。无自动安装升级环境、改batch或覆盖重启。

UCAN锁定源码、Apache2 LICENSE见vendor/manifest.json。reference_candidate.py和reference_config.json保存上一轮独立组合的参考工厂/配置；reference_pair_model.py保存原P24参考工厂。它们只用于工程对照/统一profiling，不改变实际训练工厂。
