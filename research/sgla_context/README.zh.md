# SG6 + 局部上下文 Q/K SGLA

状态：独立实现与工程验证；没有真实数据PSNR结果。基座为e731483的真实SG6，未改主干/训练器/数据/指标代码。

## 固定方案

第4个SPAB后单次写回：`Y=X+SGLA(LN(X))`。C52、1头；保留作者QKVO投影、ELU+1、全图beta、RoPE、归一化分母、V上的DW5 LePE、无额外激活的O门、输出PW。新增104通道分组DW3处理投影后的Q/K，位于ELU+1前，padding1，恒等初始化、零bias。只把原输出PW零初始化，不设第二个零gate。SG6全部门控及RGB旁路保留。

区别于原SGLA的是匹配前Q/K增加显式邻域处理；论文全文公式仍未取得，不宣称论文完全复现。训练裁剪/整图全局统计不同；普通独立tile不与整图等价。本版本仅原协议整图验证。

52/4的每头13维不兼容作者2D RoPE，故固定1头52维；不把改头数当已证实增益。DW3初始恒等时与相同权重原版SGLA的输出、输入梯度和公共参数梯度一致。

|开关 `sgla.mode`|部署参数|新增于SG6|用途|
|---|---:|---:|---|
|off / 无sgla配置|535772|0|原SG6|
|original|551008|15236|不含Q/K DW3的SGLA参照|
|context|552048|16276|本轮主候选|
|local|552172|16400|相近参数局部卷积对照|

local：LN→PW52→124→GELU→DW5→GELU→PW124→52，末PW同样零初始化。新增参数比context多124，算术量比context低约25%；不是严格等MAC。必要时另做预注册MAC匹配对照，不事后挑宽度。

用户已明确本轮硬限制为**总部署参数≤1000K**，不再使用旧B0累计10%上限。context为552.048K，满足预算；配置记录max_total_deployed_parameters=1000000，工厂强制检查。旧max_extra_deployed_parameters字段仅保留作SG6历史协议来源，不作为当前限制。延迟/显存继续实测，不设未经用户指定的淘汰阈值。SPAN训练重参数化分支及非训练eval_conv缓存的存储参数另计，1000K按Conv3XC融合后的实际推理模型统计。

## 工程验证

`verify.py --device cpu|cuda` 检查作者参考等价、奇数/非方形/1像素、SG6初值与CPU/CUDA RNG、train/eval零修正恒等、optimizer覆盖、完整网络loss反传、4步合成更新后全部新增参数实际变化、strict保存重载、FP64部署重参数化、远端导数。报告engineering-cpu/cuda.json。合成更新只验证能学习，不证明PSNR。

`integration.py` 使用原共享trainer运行两次合成更新，检查累积/缓存/1K替代夹具评测、best checkpoint、断点恢复与连续执行逐张量/损失/调度器一致。fixture输出保留在tmp和runs，不混入真实数据。

`resources.py --output ...` 在融合Conv3XC之后实测完整模型，batch1，FP32/TF32 off、20次预热、交错三轮20次CUDA计时。逐层conv MAC+解析attention matmul；未覆盖elementwise/LN/RoPE/softmax等，不冒称完整FLOPs。内存含workspace，不是纯激活。

## 真实实验协议与启动

从零seed10，初始LR2e-4，原MultiStep[200K,400K,600K,800K]、gamma.5、horizon1M，仅截停50K（50K内不衰减）；HR256、effective64/micro16、DF2K3450、L1、原增强、npy_mmap，每1K整图评测。clamp/uint8/Y/裁边4、无自集成。按DIV2K_valid最高PSNR选一个权重，统一报告六集合。没有EMA/FFT loss/预训练/专用LR。

启动前prepare会强制审计成功SG6归档、真实48K权重SHA、50K曲线、共同配置、全部有序图像hash、Python/numpy/OpenCV/scipy及CUDA环境。新候选从零，参考权重只hash不加载。不复训已有SG6，不自动跑original/local。

目标：原服务器RTX4090/torch2.2.2+cu121/Python3.11.8环境；归档：
`/home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269`。

在一个**本实验独立Git checkout**中，源码提交且干净之后：

```bash
export PYTHON_BIN=/environment/miniconda3/bin/python3
bash research/sgla_context/launch.sh FEAT-SG6-SGLA-QKDW-50K-01 \
  /home/featurize/work/span-x4-research/records/FEAT-LG6-SG6-50K-01-1792264356078917269 context
```

这是前台命令，固定只跑context；不会自动关机或释放实例。先工程检查/四臂随机推理成本，再审计原数据，随后只启动一个真实候选50K。GPU已有计算进程则退出。输出不覆盖；失败记录退出码并保留现场。没有自动续训功能，若断电应核对指纹后使用共享trainer的显式resume，而非重新运行launch覆盖。

完成自动生成 `logs/sgla_context/<group>-context/comparison-50000.json`：对SG6验证选权重、固定50K、40–50K差值趋势、逐图胜率/中位数/最差回退与图片bootstrap。单seed图片bootstrap不是跨seed证据。不会按各测试集单独best拼表；只赢B0、单次best微小变化、或局部对照同样好均不足以认定机制有效。50K灰区不自动加长。

服务器启动尚未执行，目标设备成本及数据审计仍待运行。launch训练结束（含训练失败）后将run、冻结日志/对照及源码zip复制到独立持久目录，并逐文件验证SHA256；默认归档到SG6参考归档的父目录，可用BACKUP_ROOT指定。BACKUP_COMPLETE只说明复制校验完成，还须检查training_exit_code和comparison_exit_code。不自动关机或释放实例。
