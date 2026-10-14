# EASA单组50k：复用b546036中的B0

2026-10-14。用户已明确授权新候选单独训练；不重训B0，不从B0权重初始化。B0结果源于用户服务器记录；本地已经核对b546036的实际配置与README，服务器manifest、权重hash和数据需由prepare_reuse.py在现场审核。

## 协议

源码共同部分仍为d280dc5：原SPAN、共享trainer、数据采样/缓存、评测代码未改。EASA模型仍为前轮工程验证版本。新增run.py仅放宽screen-updates参数允许50000，不压缩1M调度。prepare_reuse从归档B0 manifest复制实际配置，新增easa字段并删除moe_fs字段，唯一其他变化是本组paths_file路径。

固定seed10、从零、microbatch16、effective64（累积4次）、HR256、RGB L1、Adam2e-4、完整milestones、FP32、每1000评测保存、整图Y/crop4/uint8/无自集成，与B0一致。禁止把旧EASA模板的microbatch4直接用于本次复用。配置保留total_updates=1000000，执行停止上限50000。

现场审查：共同源码diff为空；B0原始manifest配置与b546036记录关键项一致；RTX4090/Python3.11.8/Torch2.2.2+cu121及NumPy/OpenCV/SciPy版本一致；每张HR/LR的哈希和列表顺序一致；B0完整50个评测点、验证最佳48k权重存在。权重仅计算SHA256，不加载到候选。

## 服务器入口

使用独立detach worktree固定新提交，原项目checkout和旧运行不切换分支。数据原路径来自B0 manifest；只把新增npy缓存放在本地/home/featurize/span-easa-runtime/image_cache，源码/日志/周期权重保存在work下。已有缓存也不会被本次脚本删除。

```bash
cd /home/featurize/work/span-x4-research/code/span-sr-research
git fetch origin
git worktree add --detach /home/featurize/work/span-x4-research/code/span-easa-50k origin/research/easa-mid
cd /home/featurize/work/span-x4-research/code/span-easa-50k
mkdir -p datasets /home/featurize/span-easa-runtime/image_cache
ln -s /home/featurize/span-easa-runtime/image_cache datasets/image_cache
export PYTHON_BIN=/environment/miniconda3/bin/python3
bash research/easa/launch_reuse.sh FEAT-EASA-MID-LR2E4-50K-01 /home/featurize/work/span-x4-research/records/FEAT-MOEFS-G4-LR2E4-20K-01-1791983354063378494/runs/FEAT-MOEFS-G4-LR2E4-20K-01-B0-s10
```

launch先核对B0、冻结配置、重跑目标服务器CUDA验证，再完整数据预检；全部通过才nohup启动一组EASA。不启动B0，也不调用旧MoE-FS队列。第一次哈希/解码/缓存可能较久。审查失败请保留日志并定位，不绕过数据或环境核验。不能在训练期间更新worktree源码。

进度：

```bash
tail -f logs/easa/FEAT-EASA-MID-LR2E4-50K-01/nohup.log
cat logs/easa/FEAT-EASA-MID-LR2E4-50K-01/status.txt
```

实际训练run ID：FEAT-EASA-MID-LR2E4-50K-01-E1-mid-s10。被中断后，在同一源码与配置下恢复，不再执行launch（launch会拒绝覆盖已冻结组）：

```bash
nohup /environment/miniconda3/bin/python3 -u research/easa/run.py --config logs/easa/FEAT-EASA-MID-LR2E4-50K-01/E1-config.json --id FEAT-EASA-MID-LR2E4-50K-01-E1-mid-s10 --screen --screen-updates 50000 --resume > logs/easa/FEAT-EASA-MID-LR2E4-50K-01/resume.log 2>&1 < /dev/null &
```

若手动恢复完成，需要手动生成比较报告：

```bash
python research/easa/compare_reuse.py --audit logs/easa/FEAT-EASA-MID-LR2E4-50K-01/reuse-audit.json --candidate-run runs/FEAT-EASA-MID-LR2E4-50K-01-E1-mid-s10 --output logs/easa/FEAT-EASA-MID-LR2E4-50K-01/comparison-50000.json
```

## 比较规则与证据

完成后自动产生comparison-50000.json：B0固定48k验证最佳权重，与EASA在完整50k预算中由DIV2K_valid选择的单个权重比较；所有公开集使用同一EASA权重，不能各挑测试峰值拼表。同时单列两个模型固定50k末点比较。单seed、50k短筛不代表最终收敛或跨seed稳定收益。

本地check_reuse.py三项检查通过：50k只是CLI停止cap且总调度不变；manifest路径恢复保持分组顺序和LR后缀；比较使用验证选择、最早并列权重，拒绝不足50k记录。两个Bash脚本语法检查通过。尚未访问服务器或启动此次真实训练；现场审查结果以reuse-audit.json为准。
