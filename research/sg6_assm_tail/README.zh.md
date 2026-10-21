# SG6 + ASSM：融合后幅值匹配版本

2026-10-21。版本 `SG6-ASSM-tail-RMS-v2`，独立分支 `research/sg6-assm-tail-rms`，
从固定SG6提交 `e731483d4ce4e325f3d64dbbc80add342b46f7f1` 创建。
只增加 `research/sg6_assm_tail/`，不改SG6、原始SPAN或公共训练/评测代码。

**当前仅工程原型：CPU/GPU参考路径通过；原生内核、真实图像、实际micro16和50K未验证。**
旧中段版本保留HOLD，不覆盖其配置和记录；ATD上一轮负结果也不因此改变。

## 针对旧问题的三项最小修改

旧原型在block3后，语义路由首步梯度约1e-15、FP32实际相对更新约1e-9。
两个LN并不相邻重复，删除任一没有解决路由更新问题。详见旧分支
`research/sg6-assm-mid` 的 `distribution-review.zh.md`。

本版本把候选位置移到原conv_cat之后，让修正直接经过上采样卷积进入RGB重建。
原forward完整继承，RGB门控旁路仍在上采样卷积后、PixelShuffle前相加。

```text
原SG6六块与融合 -> 原conv_cat -> F + alpha*s(F)*ASSM(LN(F))
  -> 原上采样卷积 -> 加原GB图像旁路 -> PixelShuffle
```

1. **特征分布：** `s(F)=sqrt(mean_channels(stopgrad(F)^2)+floor)`，按每个像素
   独立测量，代码是clamp_min(1e-12)后开方；只有幅值测量停止梯度，LN(F)、状态扫描和F直通
   仍有梯度。alpha保持0.01。这让归一化分支写回时恢复当地特征能量，不靠把alpha放大10倍。
   它是有意的优化设计，不是标准ASSM的一部分；没有新LN、BN、跨图统计或后置激活。
2. **优化启动：** 提示池原来由两个很小的因子相乘。现在A采用uniform ±sqrt(3/32)，
   B采用uniform ±sqrt(3/64)，对应方差1/rank和1/tokens，使两因子乘积的期望方差为1/tokens。
   同样随机流和方向，仅改变初始化幅值；未改路由温度、Adam eps、LR、loss或增加辅助损失。
   新诊断prompt RMS0.1196、原读出C RMS0.1998，在同一量级；真实图像仍待核验。
3. **归一化：** 保留输入LN和扫描后LN，各自每像素52通道，之间有投影/CPE/扫描。
   主路F、融合输出和RGB输出不新增LN。RMS测量用于乘回幅值，不再归一化主路。

没有新增任意卷积/激活，原ASSM的1x1、DW3x3、sigmoid、GELU和输出投影承担已有功能。
原图像信息没有变多，增加的是图像内语义排序聚合能力。单向扫描及参数提示池不被冒称为
每像素已读取全图其他像素当前值；SGN排序、prompt和值同序、逆排序和图间状态隔离保持。

## 同起点筛选及证据范围

`history/fix_screen.json` 保留6个同初始化反事实，而不是6次真实训练。
它在旧中段原型源码 e41f0c9acdd4539004ae759a30c8ce0670906486 的checkout根目录运行
`history/fix_screen.py`（需先创建tmp/），使用随机LR及其bicubic插值目标做一次更新。
不根据这些loss值选择PSNR赢家。

旧mid路由相对更新约8.75e-10；仅移到tail并把alpha设0.1约2.93e-5；
提示初始化匹配后约2.33e-3；tail+提示匹配+RMS恢复且alpha仍0.01约2.78e-3。
选择小alpha版本，是因为它同时缓解弱更新且写回量更保守，不是证明了全局最优。

`three-aspects-cpu.json` 和 `three-aspects-cuda-reference.json` 为独立的新版本8步检查：
- 两层路由权重每步实际相对更新约0.18%起，未再停留于仅有非零梯度的量级。
- 修正/主路RMS约0.13%；特征输入gain=.1/1/10时该比例近似保持。
- 开关分支的主干梯度方向cosine，CPU约0.99995以上、GPU参考约0.99991以上。
  仅代表同批小诊断没有观察到强烈反向干扰，不证明长期优化无冲突。
- 22个新增参数张量均实际更新，原6个空间修正分支启动；两个LN不是堆叠后失活。

8步、随机小图、bicubic合成目标不是真实SR训练，也不覆盖训练后特征分布。
状态/提示对最终RGB的依赖虽非零，初始相对完整分支作用仍较小；详见
`distribution-synthetic.json`，不能用整体分支有效冒称全局状态已成为主要收益来源。

`engineering-cpu.json`、`engineering-cuda-reference.json` 还验证了：
保留SG6初始化/RNG、非零RGB旁路、batch隔离、奇数不等边尺寸、prompt排序、
适配原码方程前后向、严格保存恢复、部署卷积融合。
本地是RTX4060Laptop/torch2.9.1；不是目标4090/torch2.2.2原生kernel证据。

## 每次版本更新的固定闸门

|方面|必须记录|不能代替什么|
|---|---|---|
|分布|F/修正RMS、偏置/幅值变化、非有限数、状态与提示作用|不能代替真实图像/训练后分布|
|优化|实际Adam更新、路由利用、主干梯度方向、optimizer覆盖|非零梯度不代表有效学习|
|归一化|层数、位置、轴、直通路径、消融证据|不能凭层数断言有害或有效|

`audit_three_aspects.py` 固化了这三项；小诊断的5%写回上限和1e-5路由相对更新下限
是本地工程筛查线，非普适稳定性/PSNR标准。每次方案变化均重新运行，训练阶段还须
在预先指定节点监测，不能靠一次8步测试保证50K不失效。

新增12,417参数，部署548,189（相对SG6+2.32%）；多了RMS测量乘法，真实延迟未测。
两层LN及参数量未增加。正式协议仍为SG6的seed10、从零、50K cap、LR2e-4、
原1M MultiStep时间轴、micro16累积4、RGB L1、每1K评测、DIV2K最佳单权重、无自集成。

## 服务器先检查内核

`check_server.sh CHECK_ID` 只检查现有环境和小图内核/三项工程诊断。
不下载数据、不安装软件、不训练50K、不释放实例。通过nohup运行即可。
缺依赖时输出NEEDS_COMPATIBLE_KERNEL并停止；内核不满足重复性时也停止，不隐藏问题。
下一步须按实际torch/CUDA/ABI在独立环境处理依赖，保护SG6环境。
原生小检查通过后，还需真实数据、micro16/最大整图显存和公共训练器集成，才能启动50K。

来源为MambaIRv2 CVPR2025，固定官方提交/许可证和明确适配差异在 `vendor/NOTICE.md`。
