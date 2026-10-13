# Featurize重训与每1000更新评测

用户选择新服务器重新训练B0与A1-G4，保留其他实验原则。首轮各5000更新，1000/2000/3000/4000/5000均评测独立验证集及五个公共测试集，保存逐图指标和checkpoint。主比较固定末3点（3k/4k/5k）；不挑测试集峰值，不自动续长训。完整1M更新学习率时间轴不压缩。

当前screen.py和launch.sh默认eval_every=1000。冻结生成配置中record_updates=1000；configs目录旧模板保留5000作为兼容参考，由入口显式覆盖。旧group不能改变间隔后强行resume；本次必须使用新group并从零跑两组。

已通过：配置冻结/模拟队列；合成两次CUDA更新，从1步恢复到2步（每步评测）与连续2步（仅末步评测）的所有模型state_dict张量逐元素一致，scheduler一致。运行ID ENGINEERING-MOEFS-a658ff7ace。该检查仅覆盖合成工程fixture，不等于DF2K长训精度或所有数据加载模式的证明。

用户提供服务器输出：RTX4090 23.54GiB、Python3.11.8、PyTorch2.2.2+cu121、CUDA可用、numpy1.26.4；缺少cv2/scipy；本地盘剩余约665GiB，同步work盘50GiB。输出仅是当时快照；尚未执行该服务器的工程检查/数据预检。

补齐依赖建议使用现有python3：numpy==1.26.4、opencv-python-headless==4.10.0.84、scipy==1.13.1；不更换torch。数据应留在/home/featurize/data的本地高速盘；大量缓存、日志和周期checkpoint同样使用本地盘，实例释放前备份重要产物到work或外部存储。数据是否迁移及实际路径仍待用户输出确认。

尚未获得服务器登录权限，也未在Featurize启动训练。下一步：用户补依赖并列出数据目录，确认私有GitHub访问，再拉取本分支、固定commit、预检、两组短筛。
