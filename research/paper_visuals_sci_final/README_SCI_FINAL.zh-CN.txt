SCI 一区实验章节补足与可视化最终套件（Tasks 1-7）
分支：research/paper-visuals-sci-final-20261128
输出目录：/home/featurize/work/span-x4-research/records/paper_visuals_20261126/step11-paper-sci-final-01

核心特性：
1. 零重训边界：仅做单次推理、历史 50K 日志提取、官方 LAM、前向特征 Hook、720p 复杂度测速与归档校验，绝不覆盖旧证据目录。
2. 新版 2022-2024 低→高方法阵营（对齐表 1，剔除过老方法）：
   - Bicubic -> SPAN52 (CVPRW'24) / LBNet (IJCAI'22) / DDistill-SR (TMM'23) / SMFANet 或 SMFANet+ (ECCV'24) / NGswin (CVPR'23)（严格按实测 5 基准均值由低到高排列）-> Ours（红字加粗）-> GT。
   - 自动复用已有已验证 PNG（仅补跑 B100/Set5 与新增 LBNet/SMFANet/SMFANet+）。
3. 全套 6 张出版级 SCI 论文图（宽 16.6 cm，300 DPI，PNG+嵌字 PDF，无图内脚注，Times New Roman）：
   - fig3a_qualitative_urban100_x4_v2（Urban100 4 建筑高频场景 × 2x4 网格）
   - fig3b_qualitative_diverse_x4_v2（Manga109/Set14/B100 4 跨数据集场景 × 2x4 网格）
   - fig4_ablation_errormap_x4_v2（6 列消融对比 + 共享 [0,40] inferno 误差热图）
   - fig_ablation_convergence_50k（50K 消融 DIV2K 验证 PSNR 收敛曲线）
   - fig5_lam_x4_v2（官方 LAM 归因 + 角标 DI + 像素贡献区域图，自动兼容纳入 NGswin）
   - fig6_feature_evolution_x4_v2（6 列：HR + Gate G + F2~F5 RMS 能量演化）
4. 任务 6：统一 1280x720 输出口径（x4 输入 1x3x180x320；x2 结构部署态输入 1x3x360x640），RTX 4090 warmup=20 / repeats=100 CUDA 同步实测，PyTorch FlopCounterMode 精确统计（1 MAC = 2 FLOPs）。
5. 双端出图省钱闭环：服务器打包 paper-visuals-sci-final-review.zip（含成图、全部 CSV/JSON 及 portable_render_assets/）；拉回本地 E:\SPANimagebase\paper-visuals-v2-final\ 后可随时在本地 Windows CPU 用原生 Times New Roman 秒级重出终稿：
   python render_sci.py all --portable-dir E:\SPANimagebase\paper-visuals-v2-final\portable_render_assets --out-dir E:\SPANimagebase\paper-visuals-v2-final
   python plot_convergence.py --logs-dir E:\SPANimagebase\paper-visuals-v2-final\portable_render_assets\ablation_logs --out-dir E:\SPANimagebase\paper-visuals-v2-final
