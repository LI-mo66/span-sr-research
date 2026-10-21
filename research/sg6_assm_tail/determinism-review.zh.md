# 原生内核重复性复审（不修改生产路径）

目标服务器用户提供的检查已存于 `server-kernel-check02.json`：
独立环境安装成功；原torch/numpy/cv2/scipy版本和原包清单未变化。
torch2.2.2+cu121、Python3.11.8、4090、ABI=false，mamba_ssm2.2.2。
输出对参考最大误差5.96e-8，参数梯度最大误差6.98e-10；
同输入重复反向的最大绝对差7.28e-12，因此严格位一致闸门停止。
这些小数值不能直接推断训练指标一定不受影响，也不是模型结构或归一化失败的证据。

官方v2.2.2反向内核有明确gpuAtomicAdd：
https://github.com/state-spaces/mamba/blob/v2.2.2/csrc/selective_scan/selective_scan_bwd_kernel.cuh
其中variable B/C跨通道归约（约305行），dD/delta_bias/dA跨batch归约（约457—467行）。
这与观察到的浮点次序差异相符，但尚未用参数级结果确认具体贡献。

`review_determinism.py` 不关掉deterministic flags、不改容差、不改模型文件。
先记录固定输入重复梯度的逐参数绝对/相对差和FP32实际Adam更新是否相同。
同时以临时执行实验验证：每张图单独调用同一原生scan，B/C显式复制到每通道，
试图使原子写入没有跨图/跨通道竞争。框架完成共享参数的归约。
这不改变前向方程、参数、LayerNorm、alpha、RMS写回或loss，但会增加内存/耗时。
只有服务器实测能说明其是否获得位一致，不能由源码推断已经成功。

每种执行检查batch2/15tokens、batch16/15tokens各8次，batch16/4096tokens检查3次。
对同初始权重做一次Adam更新的重放，不是真实数据训练，不保存候选训练权重。
报告峰值分配及耗时仅用于这个诊断，不能当完整SG6部署成本。

helper的分图/通道复制已在CPU FP64数学参考上核对：batch1/2/4、7tokens、4通道、
3状态的输出与全部输入参数梯度在atol=rtol=1e-12内一致。原生内核仍待服务器验证。

即使执行实验通过，也需独立登记执行选择，重跑模型保存恢复、三项诊断和实际micro16/
最大整图显存与延迟；不得直接把当前50K前置闸门改成放行。
本次不重新安装依赖、不改公共SG6配置、不启动50K、不自动释放实例。
