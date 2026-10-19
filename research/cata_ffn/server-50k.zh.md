# FFN对照单组50k服务器运行

独立分支research/cata-ffn-control，新增目录research/cata_ffn；旧分支和旧运行不更改。只从零训练FFN，不重训B0或完整TAB，不从任何旧训练权重启动。数据配置从B0 manifest复制；现场还要求完整TAB的共同配置、数据顺序/哈希和环境一致。

首次创建新worktree：

```bash
bash <<'BASH'
set -euo pipefail
repo=/home/featurize/work/span-x4-research/code/span-sr-research
project=/home/featurize/work/span-x4-research/code/span-cata-ffn-50k-auto
git -C "$repo" fetch origin research/cata-ffn-control
git -C "$repo" worktree add --detach "$project" origin/research/cata-ffn-control
cd "$project"
mkdir -p datasets logs /home/featurize/span-cata-ffn-runtime/image_cache
ln -s /home/featurize/span-cata-ffn-runtime/image_cache datasets/image_cache
if [[ ! -e /home/featurize/datasets/Flickr2K && ! -L /home/featurize/datasets/Flickr2K ]]; then
    test -d /home/featurize/datasets/downloads/Flickr2K/Flickr2K_HR
    test -d /home/featurize/datasets/downloads/Flickr2K/Flickr2K_LR_bicubic/X4
    ln -s /home/featurize/datasets/downloads/Flickr2K /home/featurize/datasets/Flickr2K
fi
export PYTHON_BIN=/environment/miniconda3/bin/python3
export AUTO_RELEASE=1
export BACKUP_ROOT=/home/featurize/work/span-x4-research/records
bash research/cata_ffn/launch_reuse.sh FEAT-CATA-FFN-LR2E4-50K-01 \
  "$BACKUP_ROOT/FEAT-MOEFS-G4-LR2E4-20K-01-1791983354063378494/runs/FEAT-MOEFS-G4-LR2E4-20K-01-B0-s10" \
  "$BACKUP_ROOT/FEAT-CATA-TAIL-LR2E4-50K-01-1792349126621454352/runs/FEAT-CATA-TAIL-LR2E4-50K-01-C1-tail-s10" \
  2>&1 | tee "logs/ffn-launch-$(date +%Y%m%d-%H%M%S).log"
BASH
printf '启动流程退出码：%s\n' "$?"
```

前台审核环境/数据、真实micro16显存及最大评测图、CUDA工程检查、完整数据预检，耗时较久。看到`ConvFFN only. PID=...`才可断开终端，之后nohup持续训练。不要重复执行首次创建命令，不要在运行期间修改该worktree。已有group或run会拒绝覆盖。

日志：`/home/featurize/work/span-x4-research/code/span-cata-ffn-50k-auto/logs/cata_ffn/FEAT-CATA-FFN-LR2E4-50K-01/nohup.log`。状态同目录status.txt，实际run后缀为`-F1-tail-s10`。Ctrl+C退出tail不停止任务。

训练保持seed10、DF2K3450、HR256、micro16/effective64、RGB L1、Adam2e-4、FP32及1M调度，执行cap50k，每1k评测。comparison-50000.json含验证最佳对B0、验证最佳对完整TAB，以及两个固定50k末点对照；不按测试集挑权重。

成功后将本组日志/配置/全部权重/逐图指标复制到BACKUP_ROOT下新时间戳目录；逐文件SHA256校验、BACKUP_COMPLETE及sync完成后调用`featurize instance release`。备份路径写backup-path.txt，释放结果写备份的release-status.txt。训练、比较或备份失败不释放，保留错误信息。没有启动其它候选，ATD和MambaIRv2待后续。
