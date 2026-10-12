# SPAN52近期模块文献初筛

检索日期：2026-10-12。优先2024—2026正式发表工作。用途：讨论与候选选择；不表示已在SPAN测得收益，不启动训练或改模型。

## 判断口径

先核正式发表，再核经典单图超分任务与模块消融，最后判断对SPAN52的互补性、接入成本和可复现性。排名是当前轻量预算下的验证优先级，不是实验性能排名，也不是穷尽全部文献后的全局最优。

项目基线：SPAN52、6个SPAB、×4、部署498,208参数；第一轮新增部署参数预算49,820。论文整网参数不能直接当作移植模块参数。

期刊采用JCR分区口径。TIP在2024 JCR数据、2025年6月发布的公开机构列表中为Q1；未核实中科院最新分区，因此不写“中科院一区”。

## 核心候选

|验证顺序|论文/发表|提取模块|证据与限制|SPAN接入设想|
|---|---|---|---|---|
|1|Structural Similarity-Inspired Unfolding for Lightweight Image Super-Resolution；TIP 2025，34:3861–3872|MoE-FS|作者稿表III提供有/无模块对照；原网络有展开结构及频域loss，不能把整网收益移植为保证|原四路conv_cat融合前，保留原投影；先做一处适配|
|2|SMFANet: A Lightweight Self-Modulation Feature Aggregation Network for Efficient Image Super-Resolution；ECCV 2024|EASA，SMFA为后续完整对照|正式论文表4提供×4、DIV2K训练的分支消融；完整SMFA含局部LDE，SPAN已有局部卷积，重复程度待验证|block3后单个残差修正分支；EASA独立适配属于新的候选|
|3|Emulating Self-attention with Convolution for Efficient Image Super-Resolution；ICCV 2025|ConvAttn：共享大核与输入相关动态核|有结构消融，但读取的作者稿表3是×2，且整网保留自注意力；不证明纯ConvAttn插件在SPAN×4同样有效|部分通道上下文修正；缩窄/取消共享等均需标为适配变化|
|4|CATANet: Efficient Content-Aware Token Aggregation for Lightweight Image Super-Resolution；CVPR 2025|CATA/TAB，配合IASA与IRCA|有×4组件对照；内容分组、重排与中心状态增加工程成本；作者稿表3与紧邻文字存在数字不一致|仅作为第二轮远距离内容匹配候选|

### 已核数字：全部只属于原论文设置

- SSIU作者稿表III：加入MoE-FS，Urban100 26.63→26.83、Manga109 31.29→31.60，分别+0.20/+0.31 dB；整网参数777.83K→794.47K，增加16.64K。论文采用DF2K、Y通道评测；精确训练总更新数与期刊终版表格一致性仍待复核。
- SMFANet正式论文表4：完整模型与去EASA对照，Urban100 26.19 vs 25.94、Manga109 30.72 vs 30.39；差值+0.25/+0.33 dB。这是原网络全部相关分支的消融，不能解释成SPAN单次插入的预期增益。
- ESC作者稿表3：Urban100×2，完整ESC 33.46；去大核共享33.43、同时去共享与动态核33.37、仅ConvAttn32.91。这里完整结构胜出，不支持“去掉所有自注意力更优”。
- CATANet作者稿表3：Urban100×4三行26.46/26.85/26.87，Manga109 30.81/31.24/31.31；文字却将第一阶段最大增益写为0.17，和表格不一致。本文只保留表格及矛盾记录，不据此计算模块收益排名。

## 降级与观察

- AdaIR：ICLR 2025正式论文；频率挖掘/调制用于多任务复原，主要评测去噪、去雨、去雾等，缺乏本次所需的经典×4超分直接证据。留作频率设计参考，退出首轮前三。
- SFNet/FKM：NTIRE 2025挑战报告描述直接在SPAN/SPAB后接频率模块，和骨架关系直接；但来源是CVPR Workshops挑战报告，不等于CVPR主会独立方法论文，模块独立消融尚未核实。
- IAFMNet：CVPR 2026官方活动列表出现该论文；本次全文链接读取失败，不能审核模块消融，暂不排名。
- CUST：2026作者预印本可查；正式会议页面、全文消融与代码尚未完成联合核验，暂列观察。
- SAFMN（ICCV 2023）及更早SE/ECA/CBAM：可作历史参照，不放进本轮2024—2026主名单。

## 推荐与待办

先审MoE-FS代码接口和完整成本，再比较EASA与ConvAttn的最小适配。目标是一个模块、一个位置、同预算的SPAN对照；结构比较时不同时引入论文频率loss或更改scheduler。若适配删去核心机制，则不能继续称为原模块复现。

接口约束：融合四路均为NCHW、N×52×H×W；输出仍为N×208×H×W送入原conv_cat。上下文插件输入输出均为N×52×H×W，沿LR坐标保持像素对应，dtype/device/量程与输入一致。新增分支输出应明确是修正量还是含输入的结果。池化、方差、动态核和token状态须逐项审核；当前尚未完成实现层接口审计。

文献筛选目前足以推荐验证顺序，尚不足以确定SPAN上的PSNR最优模块。需完成正式版本/官方代码审计与统一条件实验。

## 来源

1. SSIU正式出版DOI：https://doi.org/10.1109/TIP.2025.3578753
2. SSIU作者稿：https://arxiv.org/html/2506.11823v1
3. SSIU官方代码：https://github.com/eezkni/SSIU
4. TIP分区机构公开表：https://uefiscdi.gov.ro/resource-865587-JCR_2024.18.iunie.2025.pdf
5. SMFANet正式论文：https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf
6. SMFANet官方代码：https://github.com/Zheng-MJ/SMFANet
7. ESC正式论文：https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_Emulating_Self-attention_with_Convolution_for_Efficient_Image_Super-Resolution_ICCV_2025_paper.pdf
8. ESC作者稿：https://arxiv.org/html/2503.06671v1
9. ESC官方代码：https://github.com/dslisleedh/ESC
10. CATANet正式记录：https://openaccess.thecvf.com/content/CVPR2025/html/Liu_CATANet_Efficient_Content-Aware_Token_Aggregation_for_Lightweight_Image_Super-Resolution_CVPR_2025_paper.html
11. CATANet作者稿：https://arxiv.org/html/2503.06896v1
12. CATANet官方代码：https://github.com/EquationWalker/CATANet
13. AdaIR正式论文：https://proceedings.iclr.cc/paper_files/paper/2025/file/fb44a668c2d4bc984e9d6ca261262cbb-Paper-Conference.pdf
14. NTIRE2025挑战报告：https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Ren_The_Tenth_NTIRE_2025_Efficient_Super-Resolution_Challenge_Report_CVPRW_2025_paper.pdf
15. CVPR2026官方活动列表：https://cvpr.thecvf.com/Conferences/2026/Videos
16. CUST作者预印本：https://arxiv.org/abs/2607.11088

