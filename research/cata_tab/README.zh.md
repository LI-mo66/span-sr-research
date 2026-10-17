# SPAN52-ContentTAB-tail v1：工程原型

独立分支research/cata-tab，基于88e7298（共同训练源自d280dc5）。只新增research/cata_tab；未改原SPAN、EASA、MoE-FS或共同trainer/数据/指标。没有启动真实训练，工程PASS不能保证PSNR提高或全局最优。

## 接入和信息路径

首选：`原六个SPAB→原四路融合/conv_cat→ContentTAB→原上采样`。

中间接入保留为明确标注的对照配置。原型检查发现其注意力修正传到最终输出很弱；LR64×64、seed10/20/30、相同随机初始权重的配对检查支持融合后位置有更强的输出依赖与梯度，见position-audit.json。它不是三个seed训练精度实验，也不能证明该位置PSNR最优。

ContentTAB保留原TAB的CATA分组、IASA相邻子组交互、IRCA中心交互、融合投影和ConvFFN。内部已经有两次残差，直接`fused=content_tab(fused)`，不再外加fused或额外0.01gate。不叠EASA或其它候选。

参数：dim52、QK52、heads4（每头13）、MLP104、中心16、子组128、中心迭代5、EMA0.999。新增32812参数，部署498208→531020。中心832个浮点元素和bool初始化标志是buffer，不计入可学习参数。

原有主干先初始化；新增模块独立RNG seed+30001，不改变后续数据RNG。只对新增TAB Linear执行trunc_normal(std0.02)、LN权重1/bias0，Conv沿原初始化。没有加载外部预训练权重。

## 接口和衔接

|节点|尺寸与语义|
|---|---|
|输入/输出|N×52×H×W，LR像素坐标，有符号，同FP32/device，不原地改输入|
|flatten|N×HW×52，行优先；LN沿最后通道轴，不混图片统计|
|分类/排序|与16个中心的余弦相似度；离散索引不反传，Token值及投影反传|
|组内注意力|Q最多128个Token，读取本组/相邻子组最多256个K/V|
|中心注意力|共享Q读取16个中心；训练聚合microbatch，eval使用冻结中心|
|空间恢复|每图独立唯一索引，逆置换gather回原位置；不是任意reshape|
|FFN|原空间5×5 depthwise、GELU、线性投影及自身残差|
|残差输出|传给原上采样卷积/PixelShuffle；原浅层/早层/深层/T6四路融合顺序不变|

中心每个训练forward更新，microbatch16/effective64意味着每次optimizer更新4次中心更新。中心EMA不是权重EMA，不可更改microbatch而宣称同条件。buffers必须严格保存恢复。eval不更新已训练中心；初始未训练eval允许按原码从当前输入计算临时中心，不当作有效预训练推理。位置必须在config中显式写mid或tail，拒绝旧的无位置配置，避免换源码后静默改变checkpoint含义。

保留官方反射重复Token的注意力边界语义，没有宣称重复位置被mask。修复了N小于中心数，以及N刚超过子组长度时原码补齐不足的问题：按需重复反射尾部。已测N=1/7/16/127/128/129/130/191/192/255/256/257。不同中心类别可能落在同一个固定子组，不宣传为严格同类屏蔽注意力。

## 有依据的实现调整

- 将SDPA的batch和子组轴合为4D，可使用目标后端支持的高效实现；不保证本机或服务器一定启用Flash。
- 每次处理128个attention子组，减小推理score工作区。32/128/512仅作为执行分块的有限对照，不改变关系图/参数；反向可能保留全部分块，不能宣称同等比例降低训练显存。
- 对支持尺寸，用相同state_dict与固定官方源码比较输出、输入/参数梯度、中心更新；对原码失败尺寸单独验证有限反向，不把修复后的情况冒充原码等价。

来源：CATANet，CVPR2025；官方fd2222f30560d86bfac9b5f07e932c2fec12e6c1；作者稿https://arxiv.org/html/2503.06896v1。授权、参考提取与修改说明在vendor/NOTICE.md及LICENSE.txt。生产模块只依赖PyTorch；参考对照脚本另外需要einops。本方案适配完整TAB，不是复现CATANet整网。

## 本地验证

```bash
python research/cata_tab/check_reference.py --device cpu
python research/cata_tab/check_reference.py --device cuda
python research/cata_tab/verify.py --device cpu
python research/cata_tab/verify.py --device cuda
python research/cata_tab/integration.py
python research/cata_tab/resources.py --output results/cata-resource.json
```

证据见engineering.zh.md和JSON。受控远距离值依赖、全SPAN两路反传及实际更新已经分别检查，但这些不能证明已学会有用的纹理检索。真实数据瓶颈诊断、目标4090/torch2.2.2和最终PSNR仍待验证。

## 单候选入口与B0复用

没有自动训练队列，也不会自动启动ATD/MambaIRv2。configs/C1-tail-lr2e4.json为首选候选模板，C1-mid-lr2e4.json为位置对照。正式复用B0前必须用prepare_reuse.py审查实际manifest/环境/有序数据哈希并冻结C1-config.json。它只构造tail-C1，不训练或加载B0权重。

协议保持seed10、microbatch16/effective64、RGB L1、HR256、LR2e-4及1M更新调度、50k停止上限、每1k评测、Y/crop4/uint8/整图/无自集成。CATA中心是候选的新增状态，不能修改共同训练/评测。主比较选各自50k内DIV2K_valid最佳权重，所有测试共用该权重，另列固定50k末点。目标服务器工程检查和资源准入完成前，不建议直接投入50k训练。

```bash
python research/cata_tab/prepare_reuse.py --baseline-run /path/to/archived/B0-run --output-dir logs/cata_tab/UNIQUE-GROUP
python research/cata_tab/run.py --config logs/cata_tab/UNIQUE-GROUP/C1-config.json --id UNIQUE-GROUP-C1-s10 --preflight
```

训练后可用diagnose.py对真实LR/HR及验证最佳权重检查两路依赖：

```bash
python research/cata_tab/diagnose.py --config /path/to/C1-config.json --checkpoint /path/to/selected.pt --lr-image /path/to/LR.png --hr-image /path/to/HR.png --output results/cata-diagnostic.json
```

诊断报告包括attention/FFN修正幅值、中心占用、分别旁路组内/中心分支后的最终输出和指标变化。此操作不更新参数/buffer，是同权重依赖诊断，不能替代独立训练消融。不要将关闭模块后变好包装为候选成功。

## 未完成与候选提醒

待完成：目标环境验证、真实B0逐图误差/纹理诊断、单组同条件短筛、必要独立训练对照（内容分组与固定顺序/空间分组）。只有主对照有收益后才补其它位置或更多组合。

ATD自适应字典、MambaIRv2注意力状态空间仍未测试。每完成本方案的一轮，继续提醒剩余候选；不自动启动下一项。
