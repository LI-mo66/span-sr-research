# <run-id>

## 目的与状态

- 候选ID/方案版本：未填
- 假设、接入位置、唯一改动：未填
- 状态、开始/结束日期：未填
- baseline run ID及等价性说明：未填
- 结果证据类型：未填（工程检查/短筛/完整训练/多seed）

## 冻结协议

- 源码commit/dirty、配置路径/hash、环境和设备：未填
- 数据来源/hash、退化、holdout独立性：未填
- 倍率、HR/LR patch、effective/micro batch、随机seed/增强：未填
- 从零/热启动/续训、初始化与权重hash：未填
- loss、optimizer、LR完整时间轴、实际更新数：未填
- RGB/Y、range、量化、crop、整图/chop、自集成：未填
- 预先约定的checkpoint选择、固定最终/末若干轮规则：未填

## 原始证据位置

- 原始日志、manifest、逐图指标：未填
- 最新/选择/最终checkpoint及hash：未填
- 资源测量脚本、输入大小/精度/warmup/同步方式：未填

## 结果

|指标|baseline|候选|差值/说明|
|---|---|---|---|
|独立验证PSNR|未测|未测|未测|
|Set5/Set14/B100/Urban100/Manga109|未测|未测|未测|
|SSIM、逐图均值/中位数/胜率|未测|未测|未测|
|部署/可学习/总Parameter元素|未测|未测|未测|
|MAC/FLOPs及覆盖口径|未测|未测|未测|
|延迟、peak allocated/reserved|未测|未测|未测|

## 判断与下一步

失败/中断原因、是否超出基线波动、其他seed证据、协议偏差和成本取舍：未填。

结论范围与继续/停止理由：未填。不得把图片bootstrap写成训练seed稳定性，也不得把短筛最佳点写成最终增益。
