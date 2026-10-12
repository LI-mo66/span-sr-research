# 近期候选登记表

2026-10-12；当前均未获得SPAN实测PSNR增益。优先2024—2026正式发表工作；排序为验证优先级，不是性能排名。

|ID|来源|模块|状态/优先级|论文证据与迁移限制|
|---|---|---|---|---|
|A1|SSIU，IEEE TIP 2025|MoE-FS层级特征选择|首选，方案讨论|有模块独立消融；SPAN采用四路门控保留concat的适配，不等同原三路求和模块|
|A2|SMFANet，ECCV 2024主会|EASA；SMFA作完整方案备选|第二|正式论文有×4分支消融；SPAN单处插入未测|
|A3|ESC，ICCV 2025主会|ConvAttn|第三|核心结构消融为×2且依赖与自注意力配合；部分通道版本是适配|
|A4|CATANet，CVPR 2025主会|CATA/TAB、IASA、IRCA|第二轮备选|有×4消融；作者稿文字/表格不一致；分组、重排和中心状态需审计|
|R1|AdaIR，ICLR 2025|FMiM/FMoM频率挖掘和调制|降级为跨任务参考|主要证据是去噪/去雨/去雾等复原，不作为SPAN×4直接增益证据|
|R2|SFNet，NTIRE 2025挑战报告，CVPR Workshops|FKM频率知识模块|参考|直接基于SPAN但不是主会独立方法论文，模块消融待核|
|W1|IAFMNet，CVPR 2026官方活动记录|信息感知特征调制|待核|本次全文读取失败，未审消融与官方代码|
|W2|CUST，2026作者稿|相似性聚合/细节分支|待核|正式会议、全文消融、代码的联合核验未完成|

期刊分区：TIP采用2024 JCR数据、2025-06发布机构表的Q1口径；不是已核实的中科院最新分区。

## 来源链接

- SSIU：[作者稿](https://arxiv.org/html/2506.11823v1)，[正式DOI](https://doi.org/10.1109/TIP.2025.3578753)，[官方代码](https://github.com/eezkni/SSIU)。
- SMFANet：[正式论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06713.pdf)，[官方代码](https://github.com/Zheng-MJ/SMFANet)。
- ESC：[正式论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_Emulating_Self-attention_with_Convolution_for_Efficient_Image_Super-Resolution_ICCV_2025_paper.pdf)，[作者稿](https://arxiv.org/html/2503.06671v1)，[官方代码](https://github.com/dslisleedh/ESC)。
- CATANet：[正式记录](https://openaccess.thecvf.com/content/CVPR2025/html/Liu_CATANet_Efficient_Content-Aware_Token_Aggregation_for_Lightweight_Image_Super-Resolution_CVPR_2025_paper.html)，[作者稿](https://arxiv.org/html/2503.06896v1)，[官方代码](https://github.com/EquationWalker/CATANet)。
- AdaIR：[正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/fb44a668c2d4bc984e9d6ca261262cbb-Paper-Conference.pdf)，[官方代码](https://github.com/c-yn/AdaIR)。
- SFNet/FKM：[NTIRE报告](https://openaccess.thecvf.com/content/CVPR2025W/NTIRE/papers/Ren_The_Tenth_NTIRE_2025_Efficient_Super-Resolution_Challenge_Report_CVPRW_2025_paper.pdf)。
- IAFMNet：[CVPR2026官方活动](https://cvpr.thecvf.com/Conferences/2026/Videos)。
- CUST：[作者稿](https://arxiv.org/abs/2607.11088)。
- TIP分区：[机构公开表](https://uefiscdi.gov.ro/resource-865587-JCR_2024.18.iunie.2025.pdf)。

历史候选补充：SAFMN/SE/ECA/CBAM不作为本轮近期主候选；CAFT、CCNet的开源/许可证/发表状态疑点见历史candidate-cards，未进入队列。
