论文可视化补全工具 v2
日期：2026-11-27

本工具只进行推理、可视化和归档，不启动训练或微调，不修改原始模型、SG6分支或已有成果。
在独立GitHub分支和代码目录运行。默认nohup，串行占用GPU，完成后不自动关机。
必须先下载结果、本地校验并审核图3至图6，才关闭或归还实例。

本轮新增DDistill-SR和IMDN，均为作者仓库直接提供的×4权重。
DDistill-SR：icandle/DDistill-SR，固定b34bb97ba68f91a25afee23d737d97ca0a270bde；官方Inference架构和转换后权重，严格加载。
IMDN：Zheng222/IMDN，固定8f158e6a5ac9db6e5857d9159fd4a6c4214da574；官方IMDN×4架构和权重，严格加载。
所有下载文件核对Git blob，再记录SHA256；没有随机权重或本地补训对比方法。

其它候选的核查：
SRConvNet-L和ESRT：未在本次查验的官方仓库、Release和权重请求讨论中找到可用作者权重链接。
CNMC：官方README/main树未发现预训练权重指引；依赖mamba_ssm1.0.1，未更改环境安装。
LBNet：官方权重确实可直接获取。当前架构741668个参数，×4发布文件比架构多出norm.weight/norm.bias/proj.weight/proj.bias四个顶层键。本轮不自行过滤这些训练参数，保留为版本追溯任务。
用户已允许DDistill-SR搭配IMDN作为可追溯的替代组合。上述可得性只反映此次检查，不是永久结论。

默认顺序：
1. 核对原最终源码commit82296152efc423d34e8c89ad6166ae30bf01e26d、FT2 RAW和Bound8的SHA，以及此前223张图的全部旧PNG哈希。
2. 提取五个独立50K实验的原提交源码，核对共同协议、历史DIV2K选48K和权重；复制本轮用到的权重到持久化输出。
3. DDistill-SR与IMDN在Set14/Urban100/Manga109共223张图整图单次推理、逐PNG读回及同BasicSR Y通道指标。
4. 图3全部单次，Bicubic/SPAN52/SwinIR-light/NGswin/DDistill-SR/IMDN/Ours/GT八块，左整图+右2×4；无图内底部脚注。
   固定img011、img004、YumeiroCooking的既有ROI。ppt3在左侧英文文字区锁定HR(32,256,96,96)，不再用装饰图案区。
   ppt3单次局部PSNR相对最强对手不超过0.05dB时，按事先声明的工程筛选阈值换成img093，并重新对这张图按全部新增对手搜索ROI；0.05不是统计显著性标准。
   首三组原锁定ROI不会为了新增对手重新挑选。图中如局部Ours并非最好，仍保留真实分数，不篡改图像或指标。
5. 图4使用独立B0/SG6/W/U/F0历史50K选出的48K权重，单次运行img004和img011，并与历史逐图指标核对。
   误差先转float32再相减，RGB绝对误差均值，所有方法和两个场景共同使用[0,40]的inferno色标。不能替代最终1M模型的同预算消融。
6. 图5使用官方LAM固定6e75cc026ee8570f3966a6f017e9152556400876的GaussianBlurPath、grad_abs_norm和gini。
   sigma=1.2 LR像素、fold50、核5、目标16×16 HR。img004目标HR(904,472)，img011目标HR(296,920)，三个模型使用完全相同输入和目标，全部单次。
   目标与官方attr_grad同为RGB空间梯度L2幅值求和；先裁目标差分，再用vector_norm，明确零点取零次梯度。已验证非零处值/梯度与官方函数等价及平坦处有限。未直接抹掉NaN。
   逐步释放计算图，以autograd.grad只取输入梯度。原始归因、归一化图、DI和完整设置均保存。DI不能单独证明模型性能或因果机制。
7. 图6提取最终Ours真实SG6第三块有符号gate和F2/F3/F4/F5，保持Bound8。图中G为52通道有符号均值，附独立零中心色标，无额外sigmoid。
   F2至F5在同场景共享1%至99%分位数的RMS色标，另保存52通道原始张量及统计。Hook前向结果须与既有Ours单次PNG完全一致。
8. 合并新旧逐图指标，补根manifest和metrics_per_image.csv；有不同旧文件时另存-v2，不覆盖。
9. 生成小型paper-visuals-v2-review.zip；完整归档paper根目录、用到的代码/权重以及四集合HR/LR数据，逐成员读回SHA256检查。

输出：/home/featurize/work/span-x4-research/records/paper_visuals_20261126/step10-paper-suite-v2-01/
旧图3保留，新图3至6使用_v2文件名。所有图宽16.6cm、300DPI，PNG+含嵌入字体的PDF。
文本边界与重叠做自动检查；实际科学图仍须下载后人工视觉复核。本地版式测试图为合成布局夹具，明确不是实验结果，未加入服务器科学归档。

检查范围与限制：
本地PyTorch2.9.1+cu128验证了两个作者权重严格加载、真实影像奇数/矩形尺寸前向、F0+Bound8的输入梯度。
DDistill发布权重在iid随机噪声压力输入上出现过非有限值；真实img004裁剪和整图、全零及常数输入通过。没有修改模型掩盖此情况。服务器对每张真实测试图检查有限值，失败则停止，不保存无效PNG或生成虚假指标。
本地检查不能替代服务器PyTorch2.2.2+cu121/4090全库执行；历史消融逐图复现检查和实际导出图审核仍在服务器执行之后。
失败保留日志和原始产物。完成标记只表示计算和服务器备份通过，不表示本地下载已完成或可以自动关机。
