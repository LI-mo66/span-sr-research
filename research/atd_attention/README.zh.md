# SPAN52 ATD-Attention tail：工程原型

独立分支research/atd-attention，仅新增research/atd_attention。SPAN原主干、共同trainer/缓存/采样/指标和旧候选不修改。没有启动真实训练，没有PSNR增益证据。

来源：ATD CVPR2024及TPAMI2026扩展。生产适配明确依据2026官方SR源码c064084c3a33724e850ec03532fc6ddc7d7a0828，Apache2。https://github.com/CVL-UESTC/Adaptive-Token-Dictionary ，https://arxiv.org/html/2603.02581v1 。参考类、许可证和改动说明见vendor/。不是完整ATD复现，不混用2024 ADR；没有引入窗口注意力或CFFN。

## 结构与信息路径

原六个SPAB → 原四路融合conv_cat → F+TDCA(LN(F),D)+AC-MSA(QKV(LN(F)),categories) → 原上采样。F是原有融合输出，仅一次外层残差。无额外激活、gate、FFN或52→48投影。

TDCA读取梯度学习的字典，为各Token提供训练集先验；不加载外部预训练。单独TDCA不是输入图像内的跨像素信息汇聚。AC-MSA按TDCA最相似字典索引排序图像Token，每128个分组做图像Q/K/V注意力，再逆置换恢复空间，提供可能的远距离值路径。类别排序不等于任意两像素直接互连，不是严格同类别mask；同类可拆组，类别边界可混组。离散argmax/排序不可微，字典通过直接TDCA分支学习。

## 参数、接口、状态

通道52，字典128×52，TDCA QK12（单头），AC-MSA 4头/每头13，子组128。新增21813参数，总部署520021（B0 498208），约+4.38%；不是全局最优组合。

输入/输出B×52×H×W，同FP32/device/LR坐标、不原地修改。行优先展平B×HW×52；LN只沿通道。不跨图片混统计。完整反射补齐语义保留，重复位置未mask；补齐后截去填充Token并严格逆置换。字典/温度/投影/LN均是optimizer参数，无中心buffer、无EMA、无测试时参数更新。

原主干先初始化，新模块在seed+30001隔离RNG下初始化，后续CPU/CUDA随机流不变。字典N(0,1)、Linear trunc_normal .02/bias0、LN1/0、温度scale=1；2026 TDCA有效温度1+clamp(scale,0,3)ln128。没有零gate导致内部梯度启动失败的问题。

query_chunk65536只限制TDCA每次的query数，group_chunk256只限制每次AC子组数。先完成全图类别生成/排序，不能以空间切图替代全图分组。训练反向仍可能保留全部块的中间量，不宣称与推理同等显存节省。分块实现与固定官方类的输出、输入/逐参数梯度分别对照，而非仅对shape。

## 检查入口

check_reference.py：CPU/CUDA官方对照，包含极小/奇数/非整除/跨query块尺寸及FP64逐图隔离。verify.py：原主干权重/RNG、Identity旁路精确回归、尺寸范围、原数据不变、最终RGB L1两分支梯度、新参数全部真实更新、优化器无重复遗漏、序列化恢复、严格重载及Conv3XC预合并。分支旁路TDCA只关闭value输出，保留类别标签，不混同为关闭整套分类。

integration.py使用共享真实trainer和合成数据核缓存/累积/恢复/评测间隔等价；不是DF2K训练。resources.py测完整预合并FP32 B0、分块及不分块适配，交错预热/轮转20次计时、同步、同时驻留五模型；报告增量allocated及进程reserved，不冒称完整全算子FLOPs或训练后部署速度。

server_check.py只做目标环境工程检查：固定B0源码/版本、held-out有序图像哈希、官方对照、完整SPAN验证、micro16×4合成训练显存、最大评测LR整图推理、完整网络资源对比。不会训练B0/DF2K、加载已训练权重或调用释放命令；只读取B0 manifest及评测集，不需要Flickr2K训练集。

服务器命令：`python -u research/atd_attention/server_check.py --baseline-run B0_RUN --output NEW_PERSISTENT_DIR`。先用独立worktree固定提交；输出目录必须不存在。可nohup后台执行，但本轮没有自动关机；避免工程失败被误当性能失败或过早释放。目标4090/PyTorch2.2.2结果以服务器为准。

正式50k预算未开始。仍需目标环境核验，再按从零seed10、同数据/batch/LR/评测和验证最佳权重规则审查B0可比性；工程PASS不保证提高。MambaIRv2尚未测试，不自动启动。

## 本机证据与执行参数选择

CPU/CUDA官方类对照、全SPAN验证见reference-*.json和engineering-*.json。分块37/2的压力对照中输出最大误差约2.4e-7，逐参数梯度误差约1e-9量级。两分支均从最终RGB L1获得非零梯度；合成四步权重上分别旁路TDCA/AC-MSA后输出相对变化约0.0038/0.0006，仅为依赖证据，不能推断真实涨点或谁更重要。

执行参数先有限比较4k/128、16k/128、64k/256和不分块。早期engineering-resources.json及engineering-resources-grid.json中小query块延迟较大，最后独立轮转重测engineering-resources-final.json则明显下降；测试期间存在其它工程检查进程，尚不能把差异归因于确定的算子原因。保留原始报告，不挑最有利单次值宣称全局最优。

64k/256在两轮五模型对照中约19.97/19.45ms，与不分块约20.31/19.51ms接近；相应本机B0约12.51/11.92ms。以此作为暂定执行起点，而不是宣称快于原SPAN。最后一轮增量peak allocated：B0约247.5MB，64k/256约414.2MB，不分块约497.4MB；reserved为五模型共同驻留进程数据。这里只测随机初始化、LR180×320、4060Laptop/PyTorch2.9.1 FP32预合并推理，不外推4090、真实训练显存或已训练模型速度。实际trainer集成结果另行记录。

实际共享trainer集成ENGINEERING-ATD-1945d11564在64465d3通过：合成数据两步、缓存/累积、损坏日志恢复、连续与断点状态/调度器/loss记录一致，以及不同评测间隔等价。不是DF2K真实训练。

server_check整套编排在7fb72dc的本地合成manifest夹具上通过，四个子进程全部成功并生成COMPLETE。夹具使用HR32/micro1/effective2及合成评测图，未加载真实B0；其commit字段人为设置为共同基线源码以测试代码审查路径。这只验证编排/输出路径/子进程衔接，不冒充目标服务器micro16或真实数据通过。
