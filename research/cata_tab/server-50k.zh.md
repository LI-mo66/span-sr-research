# ContentTAB-tail：单组50k，nohup及完成后释放Featurize

本轮授权只训练本候选，复用b546036中的B0，不重新训练B0，不加载B0权重。不能保证PSNR提升。ATD、MambaIRv2不自动启动。

## 可直接执行的启动命令

在原Featurize服务器执行。独立worktree，不切换原checkout或其它分支。目录或实验组已存在时会拒绝覆盖，请先检查原任务状态，不重复启动。

```bash
(
set -euo pipefail
cd /home/featurize/work/span-x4-research/code/span-sr-research
git fetch origin research/cata-tab
git worktree add --detach /home/featurize/work/span-x4-research/code/span-cata-tail-50k-auto origin/research/cata-tab
cd /home/featurize/work/span-x4-research/code/span-cata-tail-50k-auto
mkdir -p datasets /home/featurize/span-cata-runtime/image_cache
ln -s /home/featurize/span-cata-runtime/image_cache datasets/image_cache
export PYTHON_BIN=/environment/miniconda3/bin/python3
export AUTO_RELEASE=1
export BACKUP_ROOT=/home/featurize/work/span-x4-research/records
bash research/cata_tab/launch_reuse.sh FEAT-CATA-TAIL-LR2E4-50K-01 \
  /home/featurize/work/span-x4-research/records/FEAT-MOEFS-G4-LR2E4-20K-01-1791983354063378494/runs/FEAT-MOEFS-G4-LR2E4-20K-01-B0-s10
)
launch_rc=$?
printf '启动流程退出码：%s\n' "$launch_rc"
```

括号将严格错误处理限制在子shell内，避免失败时退出交互终端。如果此前已经在当前交互shell执行过set -euo pipefail，先运行`set +e; set +u; set +o pipefail`恢复，再执行命令。已有worktree时不要重复执行首次创建命令，应先检查日志、进程与run状态，判断首次启动停在哪一步。

启动器在前台审计配置、版本、有序数据哈希，冻结C1-config.json；用一次性候选检查microbatch16×4反向及最大评测LR图整图推理，再运行目标CUDA工程检查与共同数据预检。一次性权重丢弃，正式训练重新seed10从零开始。此过程可能较慢；出现PID和nohup.log路径后，才表示nohup训练已启动，可以断开终端。启动前检查GPU无其它占用，本流程不停止其它任务，也不安装或升级环境。

共同协议保持HR256、LR64、micro16/effective64、RGB L1、Adam2e-4、FP32、每1k评测保存、Y/crop4/uint8/整图/无自集成。配置的调度时间轴仍1M，仅执行cap50k。新增中心按每次microbatch forward更新，不可随意降低microbatch后声称同条件。

## 进度与结果

```bash
cd /home/featurize/work/span-x4-research/code/span-cata-tail-50k-auto
tail -n 80 -f logs/cata_tab/FEAT-CATA-TAIL-LR2E4-50K-01/nohup.log
```

Ctrl+C只退出tail，不停止训练。另一个终端查看状态：

```bash
cat /home/featurize/work/span-x4-research/code/span-cata-tail-50k-auto/logs/cata_tab/FEAT-CATA-TAIL-LR2E4-50K-01/status.txt
```

训练run：`runs/FEAT-CATA-TAIL-LR2E4-50K-01-C1-tail-s10`。主报告为组日志目录的`comparison-50000.json`：B0验证最佳48k与候选50k内验证最佳比较，所有测试集统一使用各自同一个权重，另列固定50k末点。不是各测试集单独挑峰值。

## 自动关机与备份

worker顺序执行：训练成功→完整50个评测点及比较成功→复制本组日志/配置/所有周期权重/逐图指标→逐文件SHA256验证→写BACKUP_COMPLETE→sync→`featurize instance release`。这沿用此前成功的Featurize方式，不使用sudo shutdown。未访问服务器验证CLI状态；启动器会检查命令是否存在，实际请求结果写release-status.txt。

备份落在`/home/featurize/work/span-x4-research/records/FEAT-CATA-TAIL-LR2E4-50K-01-<timestamp>`；确切路径写入组日志的`backup-path.txt`。数据及npy缓存不备份，B0只引用原归档。源码版本由manifest/reuse-audit记录并已上传独立GitHub分支。备份需要容纳全部50个checkpoint以及latest/best副本，脚本会检查剩余空间。

训练、比较或备份失败时不会释放实例；会记录FAILED，避免丢失尚未核验的成果。释放命令失败时也保留失败说明，需要在平台处理。不要在运行期间更新该worktree。中断后的续训需要同源码/配置使用--resume，不能重新运行启动器；单独手动续训不会自动接续本worker的备份/释放步骤。

## 检查证据范围

本地四项检查覆盖50k停止上限不改调度、B0路径顺序恢复、统一验证最佳权重选择、备份哈希与损坏拒绝；Bash仅做语法检查，没有在本机执行释放命令。目标4090/PyTorch2.2.2环境的真实检查以服务器生成日志为准。
