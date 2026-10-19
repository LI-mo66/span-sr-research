# ConvFFN-tail 单项独立训练对照

目的：检验完整TAB的负结果是否也出现在仅增加同一ConvFFN的网络。先前同权重旁路显示注意力小幅改善当前模型PSNR、FFN移除则明显下降，但这种依赖诊断不能替代独立训练，不说明FFN本身提供数dB收益。

原SPAN四路融合/conv_cat → x + ConvFFN(LN(x)) → 原上采样。通道52，hidden104，5×5 depthwise、原GELU及内部隐藏层残差保持不变；仅一次外层FFN残差。不保留CATA/IASA/IRCA、注意力外层LN/投影、中心buffer或EMA；没有新加激活、gate或归一化。源自同一CATANet CVPR2025实现的ConvFFN，是机制对照，不是新的性能承诺。

新增参数13780，完整部署511988，较B0增加约2.77%。初始化时先在隔离RNG里构建原完整TAB，再仅保留其mlp，从而与完整TAB所有保留参数的起点逐元素相等。随后完整TAB临时对象丢弃，训练/推理只执行ConvFFN；不加载已训练B0或TAB权重。原SPAN初始化、后续CPU/CUDA随机流保持一致。初始化过程的临时构造成本不等于部署成本。

接口：N×52×H×W FP32，同设备/坐标；展平为N×HW×52，LN仅通道，FFN按原H/W恢复5×5卷积；输出恢复原NCHW。不移动像素、不按窗口补齐、不原地修改输入；没有跨图统计或持久中心状态。

共同trainer、数据/采样/缓存/指标及SPAN源码未变。B0与完整TAB旧记录现场审计：相同数据及顺序哈希、环境、micro16/effective64、seed10、HR256、RGB L1、Adam2e-4、完整1M MultiStep时间轴。实际停止50k，每1k评测；不是续训。独立选DIV2K_valid最佳权重，同一权重用于所有测试；同时比较B0和完整TAB，另列固定50k末点。

入口：`bash research/cata_ffn/launch_reuse.sh GROUP B0_RUN FULL_TAB_RUN`。PYTHON_BIN指向B0环境；AUTO_RELEASE=1时完成后备份本组所有权重/记录，逐文件校验SHA256后调用Featurize释放。训练、比较、备份失败则保留实例，不能自动视为性能失败。启动前要留意GPU是否有其它任务。

现场会重跑CUDA工程验证及micro16反向/最大评测图显存检查，再预检全部数据；通过才nohup运行一组FFN。只有进入nohup后才可断开前台。不自动训练B0、完整TAB、ATD或MambaIRv2。

本地检查：verify.py核保留FFN初值/原SPAN权重/后续RNG、Identity回归、奇数/小尺寸、真实梯度更新、优化器覆盖、严格重载和恢复、预合并；check_reuse.py核停止cap不压缩调度、路径顺序、最佳选择及备份损坏拒绝；integration.py走实际trainer合成两步及恢复。结果属于工程证据，不属于真实精度提升。

CPU/CUDA检查均PASS，详见engineering-cpu.json/engineering-cuda.json；四项复用/备份检查及Bash语法检查通过。实际trainer集成ENGINEERING-FFN-debca611bb在cac1ab6通过：合成缓存/累积/两步训练/损坏日志恢复、连续与断点模型状态及loss记录、不同评测间隔一致性均通过。本地第一次检查输出到尚未创建的tmp目录导致写报告失败，核心断言已执行；已修复输出父目录创建并完整重测通过。目标4090环境、真实数据和PSNR仍以服务器为准。

ATD、MambaIRv2仍未测试；本轮仅该对照，不自动扩展实验。
