# LG6-SiLU-MS 非线性多尺度图像旁路 v1

2026-10-17。用户批准实现。独立分支research/lg6-silu-multiscale-bypass；原B0/A/GB/PB模型与公共trainer、sampler和metrics不改。只新增候选LG6-SiLU-MS，主要比较已完成GB，同时展示PB；不重训旧方案。本轮仅工程验证，没有真实PSNR。

## 结构与接口

F=SiLU(Conv3x3[3→16](I))；U3=SiLU(DWConv3x3[16,groups16](F))；U5=SiLU(DWConv5x5[16,groups16](F))；R=Conv1x1[32→48](Concat(U3,U5))；G=2sigmoid(Conv3x3[3→1](I))；Y=PixelShuffle4(T_A+G*R)。保留GB的门控形式和RGB输入；只替换其线性R分支。

I原RGB[0,1]，N×3×H×W，旁路不减均值/乘255；A仍执行原预处理。F、U3、U5各N×16×H×W；Concat固定顺序U3/U5为N×32×H×W；R/T_A各N×48×H×W；G为N×1×H×W，仅沿48通道广播；输出N×3×4H×4W，RGB每颜色16子像素相位顺序不变。stride1，stem/DW3 padding1，DW5 padding2，最后投影padding0；均含bias。无pool/resize/空洞/BN/LN/softmax。旁路SiLU为非inplace，避免共享F被一个分支破坏；主干的原SiLU inplace及block6第一层返回特征语义保持。

相对RGB的两分支有效感受野5x5、7x7，仍是局部，不声称全局建模。非线性分支一般不能合成单个固定大核卷积；不是纯线性3x3与5x5叠加。投影无激活，允许有符号修正。G有界不等于R/总输出有界。

## 依据与假设

详见research/docs/module-study/multiscale-bypass-plan-2026-10-16.zh.md。SMFANet ECCV2024支持局部与更广上下文互补建模的动机，NTIRE2026报告支持图像旁路动机；本结构不是原模块直接复现，没有论文保证在SPAN上涨分，也不声称已经证明多尺度是瓶颈。初步实验检验整个非线性多尺度替换，包括宽度/容量变化；不能独立归因到某个因素或直接称论文创新。

## 初始化、梯度与状态

先完整构建原A，再fork_rng隔离旁路初始化，共有参数/主CPU-CUDA随机流不变。stem/DW3/DW5正常随机初始化；最终project和原RGB门控的weight/bias为0，初始R=0、G=1，整体输出与A、旧GB初始映射一致；不载入训练完权重。project首步有梯度，stem/DW3/DW5/G首步因零投影/零R暂时无梯度，后续必须真实启动；工程verify明确检查全部路径。没有再加零scale，也没有隐藏梯度裁剪/新loss/温度。

所有层仅标准parameter，没跨图缓存或新运行状态；原Conv3XC部署更新/融合逻辑保持，新非线性旁路保留而不误折叠。所有新参数进相同Adam，无分组LR。checkpoint严格state_dict重载，优化器/scheduler/Python-NumPy-Torch-CUDA随机状态沿用旧trainer恢复。

## 成本

stem448、DW3 160、DW5 416、project1584、G28，共2636；A514744→517380，比GB516116多1264（约0.245%），比PB516088多1292。相对原B0498208新增19172，server manifest的extra_deployed_parameters使用此B0口径。额外旁路卷积MACs2539HW，LR256约166.40M，不含激活/相加相乘/访存；参数小不等于延迟小，真实延迟与显存待测。

## 协议与判定

seed10、DF2K3450、HR256/LR64、有效batch64/micro16、RGB L1、Adam2e-4 betas(.9,.999) eps1e-8、MultiStep[200k,400k,600k,800k] gamma.5、FP32、npy_mmap、50k累计更新、每1000评测/编号保存。50k内不降LR，从零训练，不热启动已训练GB。最高DIV2K_valid选一个权重（同分最早），报告全部集；补充逐集峰值附checkpoint，不跨checkpoint或数据集求均值。

主要参考GB验证最高30.234560231691272 at48k。server.py核对GB完整备份中的共同配置、数据哈希和PyTorch/CUDA/GPU环境；GB备份/home/featurize/work/span-x4-research/records/FEAT-LG6-GB-50K-01-1792082028817331043。报告同时引用PB归档结果，显示delta_vs_GB和delta_vs_PB，不能只挑弱基线。单seed小幅结果不代表显著性或长期稳健。

## 服务器启动与产物

代码work独立checkout，datasets=/home/featurize/datasets；runs/logs链接/home/featurize/span-x4-runtime，备份work。Linux一键启动：

```bash
bash research/multiscale_bypass/launch.sh FEAT-LG6-MS-50K-01 /home/featurize/datasets /home/featurize/work/span-x4-research/records/FEAT-LG6-GB-50K-01-1792082028817331043
```

launch.sh检查环境、数据、已有锁、链接、源码clean，运行CUDA verify，冻结配置后nohup启动单候选。server.py共用本机GPU队列锁，依次数据哈希复核、训练、比较、绘图、SHA256备份。训练失败仍尝试保存已有产物。任何情况下不调用关机/释放，且不接受--auto-release；用户之后需明确要求才另行提供释放操作。没有该功能不意味着失败会自动停止计费。

保存每步loss_steps.jsonl，每1000的epochs.jsonl/逐图/编号权重、latest.pt和验证best_validation.pt。持久备份records/FEAT-LG6-MS-50K-01-时间编号，含logs、runs、comparison-50000.json、curves/seed-10/LG6-SiLU-MS-overview.png与SVG、backup-manifest.json、BACKUP_COMPLETE。BACKUP_COMPLETE表示文件复制校验，不等于50k训练成功，需结合status.json的exit_code与更新数。

verify.py CPU/CUDA：初始A严格恒等、共有参数/RNG、奇数矩形batch1/2、RGB相位、project首步及隐藏分支/门控后续真实学习、20步有限梯度、优化器覆盖、重载下一步一致、部署参数。integration.py：真实entry的累积/缓存/JSON恢复/优化器scheduler-RNG恢复与连续训练一致、评测频率不扰动训练、50k cap。工程通过不代表真实PSNR或长训稳定，服务器torch2.2.2与本机不同，仍须服务器verify。
