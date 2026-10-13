# B0异常区间重放

来源：用户提供的Featurize短筛记录，commit15306ee；B0第4229更新loss0.8469、4246为1081、4280峰值4397，学习率5e-4不变；A1未发生同类异常。这是训练失稳证据，尚非机制收益/失败的证明。

独立replay.py从备份epoch00004_u4000.pt恢复模型、Adam、scheduler及Python/NumPy/CPU/CUDA随机状态，只重放4001—4305。不覆盖原run，不更改LR、clip、AMP或模型，不执行自动释放。记录每个microbatch的图像/裁块/增强、输入/目标/输出范围、主要模块激活、累积梯度范数/有限性、实际参数更新。输出写持久work目录。

先严格检查原torch2.2.2+cu121、CUDA12.1与RTX4090，以及原训练PNG字节SHA256。重新建立精确uint8缓存，不制造新LR。数据因实例释放丢失，须重新上传DIV2K训练HR/LR与Flickr2K训练HR/LR；诊断不需要测试集。路径不能靠任意重采样替代。

逐步loss与原记录以rtol1e-6/atol1e-8比较，偏离时立即停止并保存诊断，避免把不匹配的轨迹当成重现。原始训练保存状态在评测后刷新派生eval_conv，因此诊断末尾也刷新这些冻结部署核；这不改变训练参数。

本地已验证裁块像素与随机流、合成基线checkpoint重放loss及所有state_dict张量/scheduler与正常训练逐元素一致；精确缓存路径复测同样通过，合成运行ENGINEERING-BASELINE-REPLAY-cbe32a69。只有原服务器同环境、同数据重放成功后才能解释真实异常。
