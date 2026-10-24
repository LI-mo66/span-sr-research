# 固定 SG6 收尾微调 A/B

从固定 commit `e731483d4ce4e325f3d64dbbc80add342b46f7f1` 新建独立分支，基座、网络定义、原训练器和旧结果均不修改。部署参数仍535772，无自集成，单次推理结构不变。只测试48K最佳权重后额外优化，不预测百万次长训收益。

## 起点与协议

生产脚本只接受完成50K、每1K记录的原SG6备份，核对source commit、配置、checkpoint/manifest指纹、备份清单SHA、完整数据HR/LR哈希和训练/评测分离、PyTorch/CUDA/GPU。固定选择验证集最佳48K权重，非50K最后权重。起点R0重新评测全部六集合，要求与归档PSNR/SSIM误差不超过1e-6，否则停止。

两个独立分支：A-L1使用原RGB L1，B-MSE使用RGB标准mean MSE；各额外10000更新。均重新创建Adam，betas/eps沿用原配置，lr1e-5，CosineAnnealingLR T_max10000 eta_min1e-6，每次optimizer更新后scheduler更新，首次实际更新使用1e-5，末次之后到1e-6。有效batch64、micro16、HR256/LR64、数据/增强不变。每分支模型加载完成后重置seed10，因此A/B使用相同裁块采样与增强流，不接着原训练的采样状态。与原训练不同之处明确为重置优化器、采样流、LR/scheduler；不能把A的收益仅归因于额外迭代。

每分支维护独立FP32 EMA，初值等于R0，从第一个optimizer.step后开始按0.999*EMA+0.001*parameter更新，不将第一次EMA覆盖为普通权重。每额外1K分别评测普通和EMA的100张验证图，保留所有快照、逐图指标和完整恢复状态。不会用公开测试集选权重。训练记录objective和共同RGB L1，不将L1与MSE原始loss数值直接对比。

各分支从普通权重的6K/7K/8K/9K/10K等权平均，仅在同一训练结构、同一分支的可训练参数上按名字处理。其余状态不能盲目平均。SPAN的Conv3XC eval_conv是派生冻结参数，必须从平均后的训练参数重新计算。当前SG6无注册buffer；脚本遇到新增buffer会拒绝并要求明确策略。EMA与平均均检查部署融合误差和参数量。

普通/EMA分别按验证集最高PSNR选权重，较早更新优先同分；两份平均权重各评测一次。最终6个候选各使用其同一权重统一评测五个公开集合。总体选择保留R0为合法候选，不强行挑较差微调权重。比较A/B相同预算、普通对普通、EMA对EMA，不能仅比较更多选择机会中的最高数值就宣称稳定收益。

## 续训与产物

中断后同组重启自动从各分支latest恢复（每1K提交），恢复optimizer/scheduler、普通/EMA参数、Python/NumPy/Torch/CUDA RNG、指标历史。未提交的step日志重建到latest，不覆盖旧基座。源码/协议/源权重/环境指纹不同拒绝恢复。成功完成A后若B失败，重启会重新校验A的产物和评测，不重新更新A。

输出在高速盘runs/<group>，完成或失败后复制校验到work/records/<group>-时间编号/results。包括manifest、R0、A-L1/B-MSE逐步日志、raw/EMA每1K权重、latest、average_6k_10k、history、selected、最终逐图结果、comparison、两组Loss/PSNR/SSIM三联PNG/SVG。队列状态和SHA清单保存在备份根；BACKUP_COMPLETE表示复制校验完成，仍需检查training_exit_code。

使用nohup，无自动关机。只运行A/B，不自动启动大patch或长训。launch.sh沿用原依赖锁定、Flickr2K链接恢复、数据及runs/logs高速盘链接和GPU队列锁。测试工程override仅可从Python内部调用并要求SG6_FINETUNE_ENGINEERING=1，生产CLI不提供短改协议入口。

微调后若要测试大patch，需要与同一起点、相同额外预算的原patch对照；本轮不执行。
