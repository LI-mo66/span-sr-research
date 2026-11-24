# 协议与服务器产物索引

日期2026-11-24。本页中的服务器路径是来源指针，不代表权重已上传GitHub。

## 来源

- 固定多倍率模型源码：0748bd60c1eaddda4e0db7f01ba768a0f2a22be3。
- 原模型F0 ×2：SG6六个52通道SPAB，UCAN在2/4块后，部分通道窗口交互在3块后，LKSA在5块后，bound8特征约束。
- 原主训练：seed10、DF2K3450对、RGB L1、FP32、LR64/HR128、有效batch64/micro16/累积4；中途引入clip0.1的经过见原1M归档。不得追溯写为从零全程固定clip。
- 普通DIV2K_valid在原1M轨迹中选出的982K：epoch00982_u982000.pt。
- SHA256：3dae5537ca672813dc4e9870f97276d173eda6f15ef8ffaf9aaf1c8406ef3068。
- 后续微调是从该权重及保存随机状态启动的新阶段，两组均重置Adam，保留原betas=(0.9,0.999)、eps=1e-8。不是直接从1M latest恢复原优化器。
- 后续阶段步号独立0–10000，lr固定，不沿原MultiStep继续下降。

## 共同评测

六集合：DIV2K_valid100、Set5 5、Set14 14、B100 100、Urban100 100、Manga109 109，共428对。×2、Y通道、裁边2、输出clamp[0,1]后round成uint8，整图推理；普通与D4×8各一份指标。×8在逆变换并平均浮点输出后量化，非八份已量化指标平均。

每个模型统一用于全部集合；不按集合换权重。后5次指标均值、某个所选权重、多个权重的参数平均分别记录。文件存在检查不等于内容检查，正式预检还审计原manifest的有序SHA256和尺寸。

## 服务器路径

项目根：/home/featurize/work/span-x4-research/code/

1. 原主训练：span-f0-x2x3-screen-01/runs/F0-X2-CLIP01-U1000K-21/。
2. LR64微调：span-f0-x2-finetune-loss-01/runs/FT-X2-982K-<arm>-10K-v1/。
3. LR96微调：span-f0-x2-finetune-patch96-01/runs/FT-X2-982K-PC96-<arm>-10K-v1/。
4. 融合评测根：span-f0-x2-weight-fusion-01/logs/f0_x2_weight_fusion/FUSION-EVAL-01/。

重点保留：

- L1-4K：LR96运行目录FT-X2-982K-PC96-l1_lr5-10K-v1/step04000.pt。
- MSE-7K：LR96运行目录FT-X2-982K-PC96-mse_lr5-10K-v1/step07000.pt。
- 当前融合候选：融合评测根/l1_mse_alpha75/deployed.pt。
- 备用50%融合：融合评测根/l1_mse_alpha50/deployed.pt。
- 融合原始汇总：comparison.json、scores.json、protocol.json、COMPLETE.json。
- 每个融合子目录12份逐图JSON和一份deployed.pt。

**当前候选deployed.pt的SHA256未在粘贴报告中输出，本归档记为未收集。** 服务器COMPLETE.json保存了验收清单；未来收集时应从该清单和实际文件核对，不以其它权重哈希替代。

已提供的LR96 latest哈希仅对应10K完整训练断点，不对应所选4K/7K或融合权重：

| 文件 | SHA256 |
|---|---|
| L1 LR96 latest.pt，10K | 56c63ff194495a7c7e229a3e736528526d99a22e9737aa22ca9375b87d5c75f1 |
| MSE LR96 latest.pt，10K | c48a0a27ba4dfee674391ca2798d7b6d34e1f1a50f7c302f13c399d94de8ca25 |

融合deployed.pt格式为F0-x2-deployed-weight-fusion-v1。须先build原模型并用deploy.fused_copy转换后再strict加载，不能直接给原训练图加载；它没有Adam/RNG，不是完整训练恢复断点。原训练断点全部保留。

## 运行中的缓存问题

旧链接分别指向/home/featurize/span-multiscale-runtime/image_cache与/home/featurize/span-f0-ft-runtime/image_cache，检查时目标均不存在；全部3450训练对和428评测对文件仍存在。该问题是缓存目录缺失，不是已证实的数据缺失。LR96创建独立/home/featurize/span-f0-ft-patch96-runtime/image_cache后运行；融合评测直接读原图，不依赖缓存。

Flickr2K实际路径/home/featurize/datasets/downloads/Flickr2K，原paths配置也是此位置；未要求另建/home/featurize/datasets/Flickr2K别名。

## 存储、关机与证据边界

实验结果写持久化work，后台管理shell放/tmp。单文件fsync/哈希验收通过且GPU空闲后才调用featurize instance release；未使用曾报NFS错误的全文件系统sync -f。融合报告明确包含CLI“实例已释放”。

evidence中的输出保留了用户复制的命令片段和结果，未编辑成服务器原始JSON。结构化结果由其中的打印表格解析，六/八位小数精度如实标记。机器验证可以核对这些转录内容的一致性，不能替代未下载的原始文件独立审计。
