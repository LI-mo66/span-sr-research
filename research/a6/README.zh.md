# A6：SPAN52 ×4 六处独立空间交互，单组50k服务器方案

只训练 `A6-Grid26-fresh-v1`，不训练B0，不加载B0权重。复用 `b546036` 中B0记录和服务器完整history；目标是检验PSNR是否改善，没有未经验证的涨分承诺。

## 模型冻结

原6个SPAB的输出分别接一个独立残差适配器：前26通道LN→center结构化全局SA→26到52投影→gamma52×修正量→加回原52通道特征。每处gamma0.01，每处独立Q/K/V与位置偏置，不共享关系或参数。保留原门控和四路融合来源/顺序，SPAB1融合支路使用增强后的输出；SPAB6内部第一卷积返回特征保留原语义。

借鉴：[HAESR commit 6943f17f544ba9adbd9375db37122c6b9d2eaa85](https://github.com/sdjvhfb/NTIRE2026_ESR_HAESR/blob/6943f17f544ba9adbd9375db37122c6b9d2eaa85/models/team06_HAESR.py) 的 GridCenterUnfold / SelfAttentionA_GridSample；不是完整HAB/HSA复现。迁移通道数、六处位置、小残差和分支zero padding是本项目待验证设计。没有HFAB、EASA、CGFN、FFT或新优化器。

NCHW，52通道进出，分支右/下补零到8倍数、输出裁回原H/W；8×8全局交错网格Q64点，KV64+49点，单头26维。中心偏移使用反射坐标索引，避免部分版本reflect pad CUDA反向不支持确定性；与独立坐标oracle的值和输入梯度检查通过。注意力不是全HW×HW。LN只作用于分支26通道，沿通道归一化；softmax沿113个KV位置。无跨图缓存、detach、EMA或持久注意力状态，训练端不混合batch统计。

新增45,438参数；训练参数2,202,574，存储参数元素2,667,402，合并部署543,646（原498,208，+9.12%）。LR64新增主要投影/矩阵乘269,528,064 MAC/图，不含softmax/LN/逐元素/索引与访存，不称全算子FLOPs。不是整个模型计算量。

## 冻结训练与B0复用

从零seed10、DF2K3450、HR256/LR64、effective batch64、micro16×累积4、RGB L1、Adam(0.9,0.999)/eps1e-8、FP32、LR2e-4。调度仍为完整1M horizon，200/400/600/800k减半；本次累计停止50k。每1k评测保存，Y/crop4、输出clamp与uint8量化、整图无自集成，完全使用原shared trainer。

原训练来源`d280dc5ebcffc46c50983dfe27e81c6096d90dcf`；新分支只增加research/a6，共同basicsr/trainer/runtime/image_io/parallel_loader必须git diff为空。prepare会核对：

- b546036中的B0有效配置、原manifest源码与环境、完整1k..50k history、验证选择48k及该权重hash、已发表的48k六数据集指标。
- 有序图像HR/LR hash与数量，禁止训练/验证重叠；读取原manifest绝对数据路径，不重新猜测数据目录。
- Python3.11.8、PyTorch2.2.2+cu121、RTX4090及原manifest TF32设置；NumPy1.26.4/OpenCV4.10.0/SciPy1.13.1。环境不符则停止，不自动升级/降级依赖。

生成A6配置与reuse-audit后，run再次检查当前配置、源码、环境和有序数据hash。原训练器的fingerprint仍覆盖源码/配置/数据/环境。baseline训练配置会被run和model工厂拒绝。工程verify会实例化随机B0用于初值/旁路等价检查，不执行真实B0训练。

## 服务器安装：优先从GitHub拉取

公开仓库分支为 `research/a6-grid`。在服务器执行：

```bash
cd /home/featurize/work/span-x4-research/code
git clone -b research/a6-grid https://github.com/LI-mo66/span-sr-research.git span-sr-a6
cd span-sr-a6
git show b546036:research/docs/module-study/results/FEAT-MOEFS-G4-LR2E4-20K-01/B0-effective-config.json
```

请使用普通完整clone，不加`--depth`或`--single-branch`，因为审计需要原训练commit和独立B0记录历史。随后按下方“只启动A6”执行launch；无需上传bundle。

## 可选离线安装：git bundle

把交付的 `span-a6-server.bundle` 上传到服务器，例如：

```
/home/featurize/work/span-x4-research/code/span-a6-server.bundle
```

下面在**新目录**克隆完整代码，原源码目录和归档B0保持原状：

bundle同时包含候选分支与独立的`evidence/b0-b546036`记录分支；B0记录不是候选分支祖先，必须保留这条历史，不能只打包候选HEAD。

```bash
cd /home/featurize/work/span-x4-research/code
git clone -b research/a6-grid ./span-a6-server.bundle ./span-sr-a6
cd /home/featurize/work/span-x4-research/code/span-sr-a6
```

复用原uint8解码缓存可节省启动时间和磁盘；这是缓存，不是训练权重。只有原目录确实存在时执行：

```bash
if [ -d /home/featurize/work/span-x4-research/code/span-sr-research/datasets/image_cache ]; then
  mkdir -p datasets
  ln -s /home/featurize/work/span-x4-research/code/span-sr-research/datasets/image_cache datasets/image_cache
fi
```

全新clone执行一次即可，若链接已存在无需重建。若不复用，原训练器会按原PNG hash重新建立同样缓存并检查空间。不要更改image_backend。

## 只启动A6

```bash
cd /home/featurize/work/span-x4-research/code/span-sr-a6
export PYTHON_BIN=/environment/miniconda3/bin/python3
bash research/a6/launch.sh FEAT-A6-GRID26-50K-01 \
  /home/featurize/work/span-x4-research/records/FEAT-MOEFS-G4-LR2E4-20K-01-1791983354063378494/runs/FEAT-MOEFS-G4-LR2E4-20K-01-B0-s10
```

解释器必须是原B0环境的Python；若实际路径不同，只调整PYTHON_BIN，不能换一套新依赖。group必须新建，不覆盖已有结果。前台依次进行B0协议/数据审计、当前服务器CUDA工程检查和数据预检，随后nohup仅训练A6。hash和全图解码需要时间，不是开始重训B0。GPU已有计算进程会拒绝启动。

进度与结果：

```bash
tail -f logs/a6/FEAT-A6-GRID26-50K-01/nohup.log
cat logs/a6/FEAT-A6-GRID26-50K-01/status.txt
```

运行目录：`runs/FEAT-A6-GRID26-50K-01-A6-s10/`。每1000更新一个记录轮；50轮=50k更新仅是本运行约定。调度不压缩成50k。

若中断，确认旧进程已结束、GPU空闲且源码/配置未改变后：

```bash
bash research/a6/resume.sh FEAT-A6-GRID26-50K-01
```

从A6自己的latest.pt恢复模型/Adam/scheduler/随机状态，累计到50k，不重新从零、不从B0权重初始化。自动修复未完成的日志尾行，源码或协议fingerprint不一致会拒绝恢复。默认checkpoint每1k，故中断可能重做最后不足1k更新。

## 完成后的比较与共享提醒

worker自动生成日志组内：

- `comparison-50000.json/md`：A6验证选择权重对B0固定48k；双方固定50k；双方48/49/50k末3点评测均值；可用时附逐图均值/中位数/胜率/最差图。
- `comparison-50000.csv/svg`：完整50个评测点的PSNR/SSIM数据与PSNR曲线。所有测试集均使用同一个DIV2K_valid选择的权重，不挑各集best。
- `reuse-audit.json`、工程日志、实际manifest和source.bundle：保留来源指纹。

**结束时终端和comparison-50000.md会提醒：先判断PSNR，再讨论是否共享注意力。** 本次没有共享；若固定点/末3点不涨，不能用共享省成本宣布成功；若有效，补重复种子和同协议fresh/shared对照。不会自动启动共享实验。

完整A6运行和日志自动备份到 `/home/featurize/work/span-x4-research/records/<group>-<timestamp>/`，逐文件SHA256校验，`backup-path.txt`给出地址。数据/缓存不复制，B0不复制不修改。备份失败会标记FAILED并保留原结果。**该入口没有关机或释放实例操作。**

本次交付未运行真实数据训练。服务器完成后提供comparison、曲线、manifest和工程日志即可分析；若缺原B0逐图文件，报告会明确标记unavailable，不伪造逐图统计。

## 工程验证与实际限制

本机CPU/CUDA原始证据在engineering-*.json。CUDA为4060 Laptop/PyTorch2.9.1+cu128，与目标4090/PyTorch2.2.2不同；launch会在目标环境重新执行verify。全模型loss反向、六分支参数真实更新、优化器覆盖、tiny/odd/non-square、主干初值与RNG、旁路精确一致、strict加载及prefusion检查已覆盖。

`resources.py`用随机合成输入测A6的micro16/累积4训练与原始eval路径，资源证据不等于涨分。原路径每次forward合并Conv3XC，不能与只算预合并路径的速度混排。RTX4060本机测得训练peak allocated约2.86GiB、reserved约3.17GiB；4090实际成本需服务器测量。可在无真实训练运行时执行：

```bash
"$PYTHON_BIN" research/a6/resources.py --output /tmp/a6-resources.json
```

`integration.py`是随机小图两步工程测试，验证真实训练入口、累积、cache、严格恢复、连续/分段结果和日志修复；不是20k/50k真实实验。`check_contracts.py`验证协议拒绝、选择规则和曲线生成。请不要把合成PSNR当模型性能。
