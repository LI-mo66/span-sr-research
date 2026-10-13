# SPAN52-FusionGate4独立短筛

2026-10-13更新：screen.py及launch.sh默认每1000更新评测/保存一次，两组一致；仍各5000更新短筛，不压缩百万更新LR时间轴。可用--eval-every 5000显式保留旧节奏；改变间隔必须新group，不能修改既有配置恢复。launch.sh第6个参数为评测间隔，默认1000。

2026-10-12。仅B0基线与A1-G4候选；原始`basicsr/archs/span_arch.py`保持不变。层级门控受SSIU/TIP2025 MoE-FS启发，采用四路加权concat而非原文三路求和。代码为独立适配。

## 目录

- model.py：FusionGate4与SPAN派生类，新增11024部署参数，总509232。
- configs/B0.json、configs/A1-G4.json：独立协议模板。
- run.py：新训练入口，复用共享训练器的采样、评测、保存/恢复逻辑；仅在当前进程替换模型工厂。
- screen.py：两组同预算短筛，冻结路径/seed/microbatch/backend配置，生成比较记录。
- verify.py：CPU/CUDA工程验证，无真实数据精度结论。
- check_queue.py：冻结配置、比较汇总与模拟工作进程退出/锁检查。
- integration.py：合成两次更新，检查累积、缓存、评测、保存和恢复。
- resources.py：复用统一资源测量口径，输出部署参数、延迟和显存。
- launch.sh、worker.sh：Linux nohup后台执行、锁、退出码记录；不调用自动关机。

方案与结果入口：`research/docs/module-study/README.zh.md`。

本地CPU/CUDA与合成两步checkpoint恢复检查通过，证据见`research/docs/module-study/engineering-moe-fs-2026-10-12.zh.md`；尚无服务器实际数据训练或4090运行证据。

## 训练协议

DF2K3450，独立DIV2K_valid100，Set5/Set14/B100/Urban100/Manga109共5/14/100/100/109对。LR必须来自已有可信bicubic×4数据，不由本脚本生成。HR256/LR64，effective batch64，默认服务器microbatch16累积4次。RGB L1、Adam、LR5e-4、完整1M更新中的200k/400k/600k/800k里程碑减半；FP32，无自集成/EMA/teacher。短筛改变停止上限，不改变scheduler。

第一阶段B0和A1各5000更新，总10000；查看结果后才同group续到每组20000，必要时再到100000。5k是学习启动/初步对照，不据此证明最终精度。默认记录周期1000；第一阶段5个评测点，比较固定末3个点（3k/4k/5k）。只以独立DIV2K_valid作后续判断；公共测试只记录。

全部HR/LR先检查数量、解码、×4尺寸、训练patch可用性；共享训练器检查train/eval HR内容hash不重叠。不能仅靠尺寸检查证明退化来源正确。

## Linux从GitHub拉取

仓库是私有的。服务器需先有GitHub访问权限，可使用`gh auth login`或已配置SSH凭据；不要把token写进URL、配置或日志。使用服务器已有CUDA PyTorch环境；所需Python包torch/numpy/opencv-python/scipy（原指标）需可导入。PyTorch2.9.1+cu128为本地验证环境，服务器版本必须先记录，不盲目升级现有CUDA栈。

```bash
cd /root/autodl-tmp
git clone --branch research/moe-fs-g4 https://github.com/LI-mo66/span-sr-research.git span-moe-fs
cd span-moe-fs
git rev-parse HEAD
python -c "import torch,cv2,numpy,scipy; print(torch.__version__,torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python research/moe_fs/verify.py --device cuda
```

`/root/autodl-tmp`是示例持久数据盘，需按实际服务器路径调整；已有同名checkout则使用新目录或确认现有分支，不覆盖未提交工作。服务器源码在首次训练后保持同commit且clean；更新文档/拉取新commit会触发恢复指纹不匹配。

## 数据目录

```text
DATA_ROOT/
  DIV2K/DIV2K_train_HR/*.png
  DIV2K/DIV2K_train_LR_bicubic/X4/*x4.png
  DIV2K/DIV2K_valid_HR/*.png
  DIV2K/DIV2K_valid_LR_bicubic/X4/*x4.png
  Flickr2K/Flickr2K_HR/*.png
  Flickr2K/Flickr2K_LR_bicubic/X4/*x4.png
  benchmark/{Set5,Set14,B100,Urban100,Manga109}/HR/*.png
  benchmark/{Set5,Set14,B100,Urban100,Manga109}/LR_bicubic/X4/*x4.png
```

如果实际目录/文件后缀不同，按`research/configs/paths.example.json`生成自己的paths文件，使用`screen.py --paths-file /absolute/paths.json`代替`--data-root`；不要靠重命名图片掩盖配对问题。自定义路径先用前台命令确认，再用标准nohup/flock运行同一命令。

GitHub只包含源码，不含数据和权重。数据需已在服务器；可使用仓库现有`research/scripts/package_server.py --data`导出本地数据包再上传，数据来源仍须核验。

## 预检与第一阶段

以下假设数据位于`/root/autodl-tmp/datasets`，唯一group为`MOEFS-G4-01`：

```bash
DATA_ROOT=/root/autodl-tmp/datasets
GROUP=MOEFS-G4-01
python research/moe_fs/screen.py --group "$GROUP" --data-root "$DATA_ROOT" --updates 5000 --micro-batch 16 --image-backend npy_mmap --preflight-only
bash research/moe_fs/launch.sh "$GROUP" "$DATA_ROOT" 5000 16 npy_mmap
```

第一轮显存预算尚未测定时可先选microbatch4或8；对照两组及后续续训必须使用相同值。npy_mmap是精确uint8解码缓存，会占额外数据盘空间；首次缓存先核磁盘空间。空间不足可在首次运行前选png，之后不能改变backend。若OOM发生在尚无checkpoint的目录，脚本拒绝自动覆盖；保留失败记录，采用新的group和更小microbatch重新跑两组。

查看：

```bash
tail -f logs/moe_fs/MOEFS-G4-01/job-*/nohup.log
cat logs/moe_fs/MOEFS-G4-01/job-*/status.txt
cat logs/moe_fs/MOEFS-G4-01/comparison-s10-u5000.json
```

launch返回PID说明进程已派发，不等于成功；以status中的exit_code=0、两组完整checkpoint与comparison文件为准。同仓库GPU锁防止重复队列；不能检测另一checkout或其他程序的全部GPU占用。程序完成不会关机，租用实例需按平台方式停止计费。

## 后续明确续到20k/100k

查看前阶段记录后分别执行，不同时启动：

```bash
bash research/moe_fs/launch.sh MOEFS-G4-01 /root/autodl-tmp/datasets 20000 16 npy_mmap
# 100k仅在20k结果讨论后执行：
bash research/moe_fs/launch.sh MOEFS-G4-01 /root/autodl-tmp/datasets 100000 16 npy_mmap
```

同group自动恢复latest.pt。默认每1000更新保存周期状态，未完成周期在中断恢复后重做。代码commit/数据hash/配置/环境必须一致；不强行跳过指纹检查。

输出：`runs/MOEFS-G4-01-B0-s10/`和`runs/MOEFS-G4-01-A1-G4-s10/`的manifest、console.log、epochs.jsonl、逐图指标、latest/best_validation/周期checkpoint；`logs/moe_fs/MOEFS-G4-01/`的冻结配置、资源报告与comparison。将这些目录备份后把小型结论填入module-study实验记录。
