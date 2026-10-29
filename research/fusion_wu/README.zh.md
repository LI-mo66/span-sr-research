# SG6 W + P24 LKSA3 双顺序短筛

固定SG6基座e731483。仅新增research/fusion_wu，F0原分支和产物不改。
F1：块2→UCAN2→块3→W→LKSA→块4→UCAN4→块5→块6。
F2：块2→UCAN2→块3→LKSA→W→块4→UCAN4→块5→块6。
两者保持W/U模块内部、独立RNG初始化、训练协议、3处分支LN、RMS适配和原骨架forward。部署均569895参数。
FUSION_ARM明确选择F1或F2；每项独立代码目录、config、run ID和备份。queue.sh串行运行，每项工程检查通过后从零seed10训练50K，再与SG6/W/U/F0比较。按DIV2K_valid最优选一个权重报告六集合，另报固定50K。
任一失败停止，不改microbatch、不跳过审计、不自动重启。两项成功且备份全部SHA256核验后，AUTO_RELEASE=1释放实例；0保留实例。其他GPU进程存在时跳过释放。
保留source-provenance.json中的原W/U固定源文件及许可证，F0为实际完成归档。单seed50K不是最优或稳定性证明，不保证涨分。
