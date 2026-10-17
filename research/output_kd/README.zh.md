# SPAN-KD-Output v1

2026-10-17。独立分支 `research/span-kd-output`。学生为原始 SPAN52×4，部署结构和参数量不变；没有 LG6、RGB 旁路、LoRA 或空间亲和损失。

## 机制与教师选择

`L = mean(abs(S(LR)-HR)) + 0.1 * mean(abs(S(LR)-T(LR)))`。

保持 B0 RGB L1；0.1 是首版设计选择，不是已证实最优值。教师用官方 SwinIR-M、DF2K、经典 bicubic×4：`001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth`，不选 GAN/真实退化模型。其任务、训练数据范围与本项目匹配，官方架构和权重可复核。HAT-S 也是合理备选，但本轮不同时更换多个教师；没有宣称 SwinIR 已被证明是 SPAN52 最优教师。DSCLoRA 的大教师负例主要涉及特征关系匹配，不能直接推广为输出蒸馏必须同构。

官方来源 https://github.com/JingyunLiang/SwinIR ，源码固定 `6545850fbf8df298df73d81f3e8cba638787c8bd`。vendor/network_swinir.py 未改动，Apache-2.0 许可证随附。教师 11,900,199 参数，仅训练时存在。学生部署 498,208 参数；教师计算和显存是新增训练成本，不能称整个实验零开销。

权重官方地址 https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth ，SHA256 `4e78e33f22c1aa8a773db0cf4a7381bae97c2362c717f155439ebc690cbd9215`。权重不上传 Git，仅下载到持久 work。严格加载 `params`，缺失/多余键或哈希不符直接失败，禁止随机教师回退。

## 信息与梯度

教师/学生同时接收同一增强后的 RGB float32 [0,1] LR，NCHW、相同像素顺序；各自执行官方内部预处理，不重复减均值。输出均为 N×3×4H×4W，未经 clamp/量化即计算全图 RGB L1。无 resize、裁剪、相位转换、掩码或插值对齐。教师支持其官方窗口 padding，输出裁回原尺寸；奇数矩形输入已纳入检查。

教师始终 eval、requires_grad=False、no_grad，仅沿 batch 以1张前向，无切图/AMP/自集成。教师参数不进入学生optimizer/checkpoint；学生全部原可训练参数正常更新，原 Conv3XC eval 缓存保持原规则。教师构造隔离 CPU RNG，学生与 B0 同seed初始化；数据采样和恢复沿用原逻辑。教师训练来源为作者发布权重，其完整训练来源未重新独立审计；官方任务/数据标签不能证明教师在每张训练patch上均更准确。

## 固定对照与产物

lr2e-4、seed10、DF2K3450、HR256、有效batch64/micro16、FP32、npy_mmap、Adam(.9,.999)/eps1e-8、原MultiStep[200k,400k,600k,800k]gamma.5，完整调度1M、执行停止50k，eval/save每1k。仅训练新候选，从零初始化，不加载B0已训练权重、不训练教师。评测量化/Y/裁边4及最高DIV2K_valid选单权重保持B0口径。

B0备份 `/home/featurize/work/span-x4-research/records/FEAT-MOEFS-G4-LR2E4-20K-01-1791983354063378494`。队列核对共同配置、数据哈希、环境、B0已完成50k和所选权重。复用B0是比较新增教师监督的整体效果，不是完全相同loss协议；不声称LoRA收益。

trainer.py 为 research/moe_fs/trainer.py 独立副本，仅加入配置校验、教师与输出loss及三项loss记录；公共trainer/sampler/metrics/架构文件不修改。教师checkpoint哈希进入冻结config和恢复指纹；换教师文件会被拒绝。断点恢复学生、optimizer、scheduler、RNG及日志；教师从固定官方权重重新加载。

每步单列 `train_l1`（对HR）、`distill_l1`、`total_loss`；overview loss图仍画对HR的L1，便于与B0比较。结束生成PNG/SVG、逐图指标、比较报告、SHA256备份和持久状态文件。备份完成不等于训练成功，需看exit_code、backup_exit_code与更新数。nohup后台；无关机/释放功能。

## 使用

先在独立checkout拉取本分支，保留其他正在运行的源码目录。服务器沿用B0的4090/PyTorch2.2.2+cu121环境。新增依赖 `timm==1.0.19`，先检查能否import；不要让安装命令升级现有torch/torchvision。如果环境缺timm，在该环境执行 `python -m pip install --no-deps timm==1.0.19`，若其依赖仍缺失应先补齐并核对环境，launcher会明确失败，不绕过检查。

`bash research/output_kd/launch.sh` 自动下载并校验教师、CUDA工程检查（含正式micro16/HR256形状）、校验B0并只排新候选。可通过前三个参数指定group/data/B0备份，TEACHER_DIR/BACKUP_ROOT/PYTHON_BIN指定持久路径和Python。默认代码work、data=/home/featurize/datasets、备份work，不使用高速盘永久保存教师。

本地验证命令：`python research/output_kd/verify.py --device cuda --teacher-checkpoint <官方权重>`；真实入口合成集成检查：`python research/output_kd/integration.py --teacher-checkpoint <官方权重>`。合成检查绝非PSNR证据，本机torch与服务器不同，启动时仍强制服务器CUDA验证。

## 已完成的工程检查

本机PyTorch2.9.1+cu128／RTX4060Laptop／timm1.0.19：官方权重严格加载、冻结与RNG隔离、奇数矩形batch、完整FP32下教师分批一致、蒸馏梯度和学生更新、序列化精确继续、498208参数部署及输出一致、正式micro16/HR256有限反向均通过。真实entry合成集成run `ENGINEERING-KD-f006ce7de5` 验证连续两步与保存恢复后第二步权重/逐步loss一致、评测频率不扰动学习、损坏日志恢复。B0哈希复用、同名多源图片、错误配置拒绝、同分最早选择、PNG/SVG及备份SHA验证通过；bash语法检查通过。详见engineering-2026-10-17.json。未在本机执行Linux flock/nohup生命周期，未运行真实数据训练。
