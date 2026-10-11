# 完整服务器套件：nohup、分批短筛与自动关机

本套件已实现独立适配模块，尚未取得候选训练PSNR增益。原SPAN源码未修改。主baseline52通道，部署498208；A融合11024，B上下文32272，C高低频3218新增参数。源机制引用及独立适配差异见candidate-cards.zh.md。实现位于research/modules/candidates.py，保持原参数初始化/数据随机流；每候选单独ID。

## 三批流程

- initial：种子10；B0、A融合、B-mid/B-tail、C-mid/C-tail共6组，先5k再20k，总计最多120000次更新（不是每组百万）。不因5k轻微落后淘汰，只停止运行失败。按固定DIV2K_valid末3个点均值排名；只临时入围PSNR正差且SSIM不下降、不同机制中最多2组。极小正差不是已稳定有效，只是继续复核入口。5个测试集只记录。
- confirm：读取同group的initial选择；种子10继续到100k、种子20从零到100k。baseline同预算复用/补齐。最多再540000次更新；没有入围者直接结束。两个种子均正差并SSIM不下降时才标为暂时幸存，不能因此宣称百万更新后稳定。
- fusion：只有confirm中2种机制独立幸存才构造一次组合；在种子10/20各跑5k→20k→100k，对比已存在的baseline和单改动；最多再200000次更新。没有2个幸存者就跳过。首轮不直接堆三个模块。

每批单独启动、单独nohup日志，批次结束自动调用关机。不会自动进入下一批，也不会自动支付最终百万长训。最大全流程累计860000次更新，运行前应结合4090吞吐评估费用；initial先支付120000次更新上限，明显比全搜索各自百万训练小。

## 环境与数据

Linux租用服务器，已可用CUDA的PyTorch/Python3.10+；numpy、opencv-python。使用平台已有torch，不盲目升级CUDA依赖。初始化入口会检查CUDA和数据目录。无自集成、无teacher/EMA/AMP，所有同批实验effective batch64；默认microbatch16，累计4次反向传播。若显存不够可在FIRST启动前改成4/8；同group后续不能改变该值或数据backend，防止不公平恢复。

路径要求：DATA_ROOT下有DIV2K、Flickr2K、benchmark，结构与本机datasets一致。配置由队列自动生成，写入logs/server_jobs/<group>/，不提交Git。组合DF2K需3450对，validation100对，五集5/14/100/100/109对。

默认npy_mmap将PNG精确解码缓存为uint8 NPY，后续随机裁块读mmap，避免每次更新反复解码64张大图。缓存按原PNG内容hash命名，原采样/像素不变；不生成另一套退化。首次缓存消耗数十GiB磁盘及CPU时间，脚本先计算可用空间；不足时停止，不能悄悄改backend。磁盘有限时第一批显式选择png，代价是吞吐可能受解码限制；以后对照全部保持同backend。缓存/转换不是训练更新。

每个候选先测统一LR180×320 FP32的原路径/预合并路径延迟、显存与部署参数；卷积MAC/FLOP字段是计数覆盖小计，明确不含functional动态卷积、池化和逐点运算，不冒充完整FLOPs。模型失败日志独立保留。

## 上传代码和本地数据（无需服务器登录GitHub）

在本机PowerShell项目目录运行：

```powershell
cd E:\SPANimagebase\span-research
E:\anaconda\envs\dl\python.exe research/scripts/package_server.py --data
```

生成server-upload/span-source.bundle（全部Git历史）、df2k-benchmarks-x4.tar（仅所需PNG，排除旧下载包/缓存/×2×3）及manifest.json。重复导出时不覆盖已有bundle，先保留/重命名旧导出。GitHub私有仓库已包含所有源码；bundle方案避免服务器配置GitHub凭据。

先在平台网页终端执行`mkdir -p /root/autodl-tmp`（该目录需实际确认是持久数据盘），然后本机传输：

```powershell
scp -P 12799 E:\SPANimagebase\span-research\server-upload\span-source.bundle E:\SPANimagebase\span-research\server-upload\df2k-benchmarks-x4.tar E:\SPANimagebase\span-research\server-upload\manifest.json root@connect.nmb1.seetacloud.com:/root/autodl-tmp/
```

也可通过平台文件上传界面传三个文件。第一次代码迁移在服务器执行：

```bash
cd /root/autodl-tmp
git clone -b research/main span-source.bundle span-sr-research
cd span-sr-research
git remote set-url origin https://github.com/LI-mo66/span-sr-research.git
mkdir -p datasets
tar -xf /root/autodl-tmp/df2k-benchmarks-x4.tar -C datasets
python -c "import torch,cv2,numpy; print(torch.__version__,torch.cuda.is_available())"
```

如项目目录已存在，不覆盖；改为独立新目录或使用已有相同commit checkout。数据包自生成于本机PNG目录，没有外部任意成员路径；传输后可用sha256sum与manifest校验。原官方权重不必上传，因为主短筛从零训练。

## nohup启动与自动关机

在服务器项目根目录，使用自己的唯一group，例如SPAN52-SCREEN-01：

```bash
bash research/scripts/launch_nohup.sh SPAN52-SCREEN-01 "$PWD/datasets" initial 16 npy_mmap
```

launch_nohup内部执行nohup并重定向stdin/stdout/stderr，返回PID和日志路径，断开SSH不影响后台。**正常完成与任务报错都会在写入status.txt和sync后调用/usr/bin/shutdown，防止失败后继续计费。** 路径可由SHUTDOWN_BIN覆盖，必须是平台已有可执行程序。本平台当前SSH不可访问，实际关机效果尚未在此租用实例验证；已根据AutoDL官方文档的/usr/bin/shutdown调用方式实现，不将跨平台验证当作已完成。不要在已有其他需要持续运行工作的实例启动这批自动关机任务。

观察日志（日志路径也会在启动时打印）：

```bash
tail -f logs/server_jobs/SPAN52-SCREEN-01/initial-*/nohup.log
```

关机后重新开机查看：

```bash
cat logs/server_jobs/SPAN52-SCREEN-01/initial-*/status.txt
cat logs/server_jobs/SPAN52-SCREEN-01/ranking-s10-u20000.json
cat logs/server_jobs/SPAN52-SCREEN-01/selected-initial.json
```

同group再次执行initial会从已有latest.pt继续，不从零覆盖；训练前要求同commit/配置/数据hash/环境，latest写入原子化。恢复仅保留已完成5000更新周期，未完成周期重做。确认复核/融合时分别启动：

```bash
bash research/scripts/launch_nohup.sh SPAN52-SCREEN-01 "$PWD/datasets" confirm 16 npy_mmap
bash research/scripts/launch_nohup.sh SPAN52-SCREEN-01 "$PWD/datasets" fusion 16 npy_mmap
```

两行应在相应前批完成并重新开机、查看结果后分别执行，不同时启动。仓库内flock防止同项目重复GPU队列；未获得锁的任务拒绝启动且不会关掉锁持有者。退出被SIGTERM时同样记录并关机；SIGKILL、系统崩溃无法执行EXIT trap，不能保证程序级关机处理覆盖这类外部中断。

每组结果runs/<group>-<variant>-s<seed>，包含逐图指标、epochs.jsonl（平均loss/五集/独立验证best及epoch）、console.log、环境/数据/代码manifest、每5000更新完整checkpoint、latest/best_validation；每批总体日志与失败/排名在logs/server_jobs/<group>/。数据/大型权重/日志不提交Git。

## 最终百万长训

本套件不自动启动最终长训。选定结构后，baseline及候选在各自固定配置/commit中续到1M或按预注册种子独立从零训练；不能只长训候选。不同代码更新/环境改变不能直接跨指纹resume。正式论文指标统一验证选权重或固定末点，保留所有失败结果。
