# SPAN52 B0 versus SG6+W+UCAN24+LKSA5+feature-bound8, x2/x3

固定基座 e731483d4ce4e325f3d64dbbc80add342b46f7f1，新增文件仅在本目录。原始SPAN和SG6文件不修改。

候选六个SG6块的主输出按绝对值8为不变区间、16为平滑上界；LKSA残差写回和相加后的输出采用相同约束。与x4服务器bound8恢复方案的约束位置一致。W、UCAN的位置、窗口、通道、初始化不改。

GB旁路输出由48改为3*r*r；重建头使用r倍PixelShuffle；裁剪、对齐、x8逆变换和裁边均使用对应倍率。灰度PNG用IMREAD_COLOR读取，不修改原文件。x3 HR先按LR尺寸乘3在右侧/底部对齐，再裁边3。

四组独立从零：x2 B0、x2 F0、x3 B0、x3 F0。每组seed10、50K更新，LR64、HR分别128/192，micro16*accum4=64，RGB L1、FP32、Adam2e-4；MultiStep200K/400K/600K/800K，50K内不减半。候选与同倍率B0共享数据与训练、评测协议。

每1K保存/评测，普通与D4x8同时报告。按普通DIV2K_valid最高PSNR选择一个权重、最早并列，统一报告所有集合，x8为同权重的次要指标。不得按各测试集分别挑权重。短筛对照不等于同预算长训对照。

执行顺序：数据完整解码/有序SHA/重叠检查；CPU/CUDA模型检查；真实入口2+2与连续4步严格恢复（模型/Adam/调度器/所有RNG），指标和灰度/非整除裁剪检查；全部四组micro16*accum4与最大图像x8检查；四组串行50K；每组SHA校验持久化备份。不自动关机。

本地工程验证使用torch2.9.1和4060，服务器必须用Python3.11.8/torch2.2.2+cu121/4090重新验收。工程通过不保证PSNR上涨或永不出现长期异常。

运行：`bash research/multiscale_f0/launch.sh`。仅验收：`CHECK_ONLY=1 bash research/multiscale_f0/launch.sh`；该模式有意不自动提供原目录二次启动，防止覆盖数据与记录。

来源：基础LG/GB/SG6来自固定基座；attention/lksa/window来自F0快照，保留UCAN许可。上游出处见UCAN-source-manifest.json。trainer/ensemble/guard来自x4长训开发版本82296152efc423d34e8c89ad6166ae30bf01e26d并作本倍率适配。
