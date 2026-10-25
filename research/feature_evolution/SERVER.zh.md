# SG6-FE两方案服务器交付

2026-10-25。GitHub分支research/sg6-feature-evolution，固定SG6来源e731483。
默认串行：加法版50K -> 差分版50K；各seed10从零。原SG6只读复用，不参与训练。
正式协议保持Adam2e-4、L1、HR256、micro16/effective64、原1M MultiStep时间轴，每1K评测六集合；50K截停。
两组均按各自DIV2K_valid最高PSNR选单权重，六集同权重报告；另列末点/末3/末10及逐图统计，差分对加法单独比较。

## 默认路径

- Python：/environment/miniconda3/bin/python3
- 数据：/home/featurize/datasets
- SG6归档：/home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269
- 持久输出：/home/featurize/work/sg6-fe-records/SG6-FE-时间-PID
- 高速盘运行：/home/featurize/span-sg6-fe-runtime/时间-PID
- 缓存：/home/featurize/datasets/image_cache

若路径不同，在调用launch.sh之前设置PYTHON_BIN/DATA_ROOT/REFERENCE_BACKUP/OUTPUT_ROOT/CACHE_ROOT。
使用独立、干净的新clone；禁止在已有训练目录checkout或覆盖。启动脚本拒绝已有runs/logs/datasets目录，保留旧实验。
锁与历史队列统一用/home/featurize/span-x4-runtime/logs/server-gpu.lock。

## 启动

先从GitHub拉取本次交付的固定commit，在Linux Bash中运行：

```bash
bash -n research/feature_evolution/launch.sh
bash research/feature_evolution/launch.sh
```

launch.sh使用nohup在后台提交完整server.py队列，退出SSH后继续运行。前台打印PID、GROUP和nohup.log路径。
不要再在外面nohup一次；不要同时启动两份队列。

## 预检与观察

后台队列先检查GPU空闲、干净源码、关机权限和持久输出空间（至少12GiB）；再核对固定SG6完整50K历史、配置、48000所选权重已知SHA256、有序HR/LR哈希、环境与评测口径。
原环境要求Python3.11.8、torch2.2.2+cu121、RTX4090、numpy1.26.4、cv2 4.10.0、scipy1.13.1；依赖不符直接停止，不自动升级或安装。
候选入口显式沿用审计出的SG6 TF32策略；单独工程验证关闭TF32仅为数值诊断，不能写成生产策略改变。
再运行CUDA工程验证、两组真实训练入口的两步合成恢复验证、真实数据解码预检及小尺寸推理剖析。全部完成后才从零训练。

新增Observer在前12次与每1000次更新采样：a/b/c阶段、融合delta、两路门控的shape/RMS/mean/absmax，及新增参数实际更新L2。
观察器不改变张量/梯度/optimizer/RNG；本地三步CPU等价测试以及带观察器的入口恢复测试单列归档。
采样只覆盖部分步骤，且不证明没有任务梯度竞争或PSNR必然提高；不自动添加norm、裁剪或改学习率。
队列不自动续训。若失败保留完整代码、日志与断点，需要先审查原因；不要自行只改run ID重新覆盖或改变冻结配置。

## 输出与自动关机

每组完整权重、optimizer、scheduler、RNG、每1K断点、best_validation、逐步loss、逐图指标均复制到GROUP/{add,difference}-backup。
每个文件按SHA256和大小验证；backup-manifest.json内training_exit_code与BACKUP_COMPLETE必须一起看。
GROUP内有SG6来源审计、环境/工程预检、真实特征观察、两组对基座比较、两组互比、曲线和源码tar.gz。
可回传SG6-FE-RETURN.zip做结果审查；它包含日志/指标/源码/曲线，不含权重和数据。完整权重留在上述备份目录。

两组运行完成不代表PSNR通过；即使候选不涨点，正常完成、对照与备份完整后仍按用户授权关机。
关机严格要求两组状态COMPLETE、训练退出0、完整备份重校验、回传ZIP哈希验证、GPU没有其他计算任务。
满足条件后执行系统shutdown -h now（非root用sudo -n），不调用云实例销毁/释放API。
权限提前用sudo -n -l检查；若平台只允许其专用释放命令，当前脚本会在训练前停止，不擅自改成销毁实例。
预检/训练/对比/备份/打包任一失败，停止队列并保留服务器，查看failure.txt与FINAL_STATUS.json。
系统关机实际是否执行及计费规则由平台决定；COMMAND_ACCEPTED只表示命令接受，不能作为实例已销毁的证据。

## 本地验证范围

delivery-checks.json覆盖不改变模型更新与RNG的Observer、SHA备份/回传ZIP、合成对比末点/末3/末10，以及mock关机保护。
不会在本地执行实际关机；服务器权限/真实数据/运行成本须由后台预检确认。
原模型CPU/CUDA合成工程证据见engineering-review.zh.md；本轮新增入口后另跑带Observer的两组恢复检查，记录见server-delivery-review.zh.md。
