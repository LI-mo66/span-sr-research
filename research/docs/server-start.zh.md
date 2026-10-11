# 租用4090服务器：连接与启动准备

用户提供：ssh -p12799 root@connect.nmb1.seetacloud.com。连接可达，当前本机公钥未被接受。不存在本机读取到的已可用登录权限，不索取聊天中的密码/私钥。

请将research/scripts/authorize_existing_public_key.sh内容复制到租用平台的root网页终端执行。脚本只添加本机现有公钥、保留原有authorized_keys，不修改sshd配置。密钥公钥指纹：SHA256:U27YCGAmtC1EaVs6rYUKKy4a20dsv/6GXZVyAh6gQgE。随后原SSH命令可通过已有本机私钥认证；私钥仍留在本机。

连通后的顺序：检查pwd/GPU/磁盘/CPU/已有Python环境；选用平台持久训练盘，若存在/root/autodl-tmp通常考虑放在那里，但实际路径必须核验，项目目录暂未知。克隆私有研究仓库到独立目录；服务器GitHub认证由已有授权或SSH方式处理，不向日志复制token。数据和大型权重用可续传传输，核对文件hash/图像对；本机paths.local.json不能原样作为服务器绝对路径配置。

测量microbatch4/8/16/32/64的FP32更新成本与显存，effective batch始终64。用research/scripts/benchmark_training.py可测计算部分；它不包括数据加载，不能直接声称完整训练ETA。正式baseline启动还要核实时数据吞吐/裁块随机流，必要的CPU并行或缓存优化对baseline与候选统一应用，记录为协议变更，不单独加速候选。

例如在服务器项目根目录（python为已核验的训练环境）执行：

```bash
python research/scripts/benchmark_training.py --micro-batch 16 --output results/train-speed-m16.json
python research/train.py --id B0-SPAN52-x4-s10-server-unique --preflight
python research/train.py --id B0-SPAN52-x4-s10-server-unique --screen --screen-updates 5000
python research/train.py --id B0-SPAN52-x4-s10-server-unique --screen --screen-updates 20000 --resume
python research/train.py --id B0-SPAN52-x4-s10-server-unique --screen --screen-updates 100000 --resume
python research/train.py --id B0-SPAN52-x4-s10-server-unique --resume
```

所有阶段同一冻结配置/数据/源码，最终resume不带--screen继续至1M；仅改变运行停止上限，保持相同LR时间轴。源代码指纹检查严格，启动后不能一边修改训练源码/配置一边声称同协议续训。主baseline模型未改变；候选使用独立分支/ID。服务器长时间后台运行与日志写入均属原实验授权范围，具体执行在连接/预检通过后进行，当前没有训练进程在这台远端由本项目启动。
