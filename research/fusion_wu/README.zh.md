# SG6 + W + P24/LKSA5 首轮融合

仅一个候选，从零seed10训练50K。固定基座e731483d4ce4e325f3d64dbbc80add342b46f7f1。
W源78cc1dda90a5bafcefa6c62225197f6eb95c92e2；U源d42b73fef5b2ec3e088f340ab033a5c4ac13a190。
在块2、3、4、5主输出后分别加入UCAN、W、UCAN、LKSA。保留原forward、first/gate返回、融合头和GB图像旁路。
W/U权重不用于正式训练初始化。新增模块的各自RNG种子与原版本相同；零U输出下融合初始映射等于W，而非严格SG6。
三处分支LayerNorm；LKSA按图像RMS输入归一化并恢复；不新增归一化、损失、激活、学习率策略。
部署569895参数；训练重参数化模型的原始参数量不同，不能混用两者。

发布分支直接基于固定SG6，只修改research/fusion_wu。W/U源文件已按固定远端commit的Git blob SHA逐项核验，记录于source-provenance.json，服务器再次核验。launch_reuse.sh使用nohup后台启动worker_reuse.sh；不在服务器生成或修改源码，不切换原目录分支。
自动步骤：参考/环境/数据审计，CPU/CUDA接口与真实参数更新，真实训练入口两步恢复一致性，真实micro16累积4显存检查，最大验证图整图推理，再进行50K正式训练、统一权重比较及SHA256备份。
任何工程失败停止，不自动训练或关机；正式屏筛不允许通过本发布脚本自动resume。
自动关机默认且固定关闭，沿用用户最近要求。

W/U来源的Apache-2.0代码保留源LICENSE与清单；源码注释已明确来源，未将其宣称原创。
单seed50K结果不能证明最优、跨seed稳定或最终收敛；若融合未同时超过两单独模型，保留原候选，不自动调参抢救。
