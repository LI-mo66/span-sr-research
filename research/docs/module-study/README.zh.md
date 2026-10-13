# SPAN模块研究：候选、方案与实验档案

创建日期：2026-10-12。当前状态：文献筛选与方案讨论；未启动新训练。

2026-10-13更新：[Featurize重训/每1000更新评测](featurize-eval1k-2026-10-13.zh.md)。新服务器B0/A1从零、各5k短筛；实际服务器尚未启动。

本目录作为后续讨论和实验结果的统一入口：

- `candidates.zh.md`：全部近期候选、文献来源、证据等级和最新优先级。
- `moe-fs-plan.zh.md`：首个候选的尺寸、融合、参数和待确认设计。
- `experiments.zh.md`：实验索引与结果汇总，包含失败和无增益结果。
- `experiment-template.zh.md`：每次实际实验复制为`records/<run-id>.md`后填写。
- `parameter-count-2026-10-12.json`：本次实际模型构造参数计数；不是训练或PSNR实验。
- `engineering-moe-fs-2026-10-12.zh.md`：新独立实现的CPU/CUDA、队列和合成训练恢复检查证据。

首个已确认候选的独立源码和服务器命令：`../../moe_fs/README.zh.md`；Git分支`research/moe-fs-g4`。训练期间服务器checkout保持源码commit不变，本目录结果在本地追加后另行提交，不直接改变正在训练的服务器指纹。

详细检索报告已复制为本目录`literature-screen.zh.md`并随代码上传；工作区根目录保留原文件。此前`../candidate-cards.zh.md`为2026-10-11历史设计；特别是AdaIR排序已被本次文献筛选更新，不应直接按旧排序启动队列。

原始日志、权重、逐图指标仍保存在仓库`runs/<run-id>/`，本目录保存可审查的小型记录及对应路径/hash。`runs`被Git忽略，后续需独立备份；Markdown汇总不能代替原始数据。训练源码和配置指纹不会为了更新文档而自动忽略差异。

追加规则：每项结果写清证据类型、run ID、模型方案版本、baseline匹配关系、协议、checkpoint规则、数据/代码hash和成本。空白写“未测”，不写0。论文收益与本项目收益分列；保留失败记录，不覆盖旧结论。训练前单独冻结方案和完整协议。
