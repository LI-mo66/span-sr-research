# SG6-UCAN独立版v2服务器交付

分支 `research/sg6-ucan-qk-pair`，直接从固定SG6 commit e731483建立。
唯一真实实验ID `FEAT-SG6-UCAN-IND-V2-50K-01`；部署554128参数。
所有候选与服务器代码只在本目录，不修改原SPAN、SG6、trainer或既有产物。

本地已完成CPU/CUDA机制、尺寸、优化启动、跨块梯度和真实入口合成续训验证。
交付新增脚本另测试路径顺序/数据备用目录、验证选权及并列规则、同权重六集
报告、不完整历史拒绝、失败备份哈希和拒绝覆盖；launch.sh经过Bash语法检查。
Linux控制器尚未在用户服务器执行，不把本地检查写成远程运行成功。

## 用户执行流程

在新的源码目录、独立高速runtime下，用当前GitHub提交启动：

```bash
export PYTHON_BIN=/environment/miniconda3/bin/python3
bash research/ucan_qk_pair/launch.sh FEAT-SG6-UCAN-IND-V2-50K-01 \
  /home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269
```

完整复制命令由本次交付消息给出，包含新目录、固定提交、独立高速runs/logs/cache。
不checkout已有工作目录，不改数据软链接；原数据别名不存在时，只在本实验
paths.json中指向已存在的downloads/Flickr2K，仍核对逐图顺序与内容哈希。

launch.sh只提交一个nohup后台控制器，立即返回PID/log/status路径。
断开终端不会中断后台任务。看到PID不代表审计或训练已经通过，应看status/log。

## 必须通过的服务器检查

1. 全局GPU文件锁和nvidia-smi计算进程检查，源码干净且保护文件无差异。
2. SG6 BACKUP_COMPLETE、training_exit_code=0、唯一实际SG6 run、固定来源commit、
   完整配置、48K所选指标与checkpoint SHA256核对；不加载SG6权重训练候选。
3. 参考环境一致：CUDA/PyTorch/GPU/TF32标记与SG6 manifest相同，参考Python3.11.8、
   numpy1.26.4、OpenCV4.10.0、scipy1.13.1。缺依赖或不一致时停止，不自动安装升级。
4. 3450训练对及100/5/14/100/100/109验证对，按SG6训练列表顺序逐图SHA256核对。
5. fast runs/cache和持久备份各至少8GiB可用；实际图片缓存额外有原trainer空间检查。
6. 在服务器实际环境运行verify.py、真实trainer入口integration.py合成续训，
   再测完整模型成本。任一失败不启动真实数据50K。

之后才执行从零50K：seed10、LR2e-4、micro16/有效64、HR256、L1，
原MultiStep完整1M时间轴，每1K评测，不用自集成，原Y/量化/crop4不变。
生产入口明确设为SG6归档TF32标记（cudnn=True、matmul=False）。

## 输出、比较与备份

- 高频产物：独立runtime的runs/logs/image_cache，保护已有runtime与缓存。
- 状态：`logs/ucan_qk_pair/FEAT-SG6-UCAN-IND-V2-50K-01/status.json`。
- 固定配置/数据路径/来源审计/服务器工程报告在同目录frozen/。
- 原trainer保留每1K权重、latest、best_validation、逐步loss与六集合逐图指标。
- 50K后只按DIV2K_valid最高PSNR（并列选最早）选一个权重报告六集合；另列50K
  末点、40--50K验证曲线差、逐图差与图片bootstrap，不把它当跨seed稳定性。
- 绘制原始loss、DIV2K验证PSNR/SSIM曲线。
- 完成或失败后复制到持久work/records的新目录，逐文件SHA256校验，附source.zip。
  不覆盖旧备份；最终状态另存FINAL_STATUS.json和持久status副本。

`BACKUP_COMPLETE`只表示复制校验完成；成功还必须满足最终status.success=true、
training_exit_code=0、comparison_exit_code=0、plotting_exit_code=0、backup_exit_code=0。
null或-1表示相应步骤没有执行。失败现场保留，不自动续训、不静默重启。

不自动关机或释放实例，不启动共享版、SG6重训或长训。不含SSH远程执行代码。
共享版只保留工程机制验证接口，若后续获准训练，另从固定基座建立独立分支/目录。
