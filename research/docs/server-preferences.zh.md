# 服务器操作偏好

2026-10-17用户明确要求：后续实验采用nohup后台运行，结束后自动绘图并备份保存到持久work；不自动关机/释放，只有用户另行明确说明才启用。此要求取代先前实验默认auto-release设置；不要从旧启动命令继承自动释放。

数据仍为/home/featurize/datasets，本地高速运行产物与日志经链接保存；源码与持久备份在/home/featurize/work/span-x4-research。实例释放会清除高速盘；BACKUP_COMPLETE只表示备份复制校验，训练成功需看退出状态和最终更新数。
