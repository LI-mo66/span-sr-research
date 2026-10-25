# 服务器交付检查

2026-10-25。交付的是可供用户执行的Linux服务器队列，未在本会话连接服务器或启动真实数据训练，也未执行实际关机。

- 原SG6、共享trainer、数据/指标/loss/调度器文件无改动；相对e731483全部变化位于research/feature_evolution/。
- launch.sh已通过实际Bash的bash -n语法检查（仅解析，没有执行启动或关机）。首次受限MSYS进程无法创建系统命名对象，随后仅对只读语法检查使用获准环境，语法PASS。
- 新增Python文件compileall通过，git diff --check通过。
- delivery-checks.json：Observer三步梯度累积检查中，模型state和RNG逐bit不变；SHA备份、轻量ZIP、合成指标对比与关机保护mock测试通过。测试没有调用真实shutdown。
- cb7cf3ff71fce2966608a0f57da98829b930ff82下，两组启用FE_OBSERVE的真实入口合成恢复检查PASS：加法ENGINEERING-FE-1456330cde；差分ENGINEERING-FE-674f49be8b。日志已保留为server-integration-add.txt和server-integration-difference.txt。
- 随后修正观察器恢复时的更新编号，源码2fd58b7（完整hash可见本地Git）；再次运行差分入口恢复检查ENGINEERING-FE-88423cee82，连续/恢复参数一致，观察日志actual_update编号严格为[1,2]，没有重新从1标号。记录见server-integration-resume-offset.txt。
- 原模型CPU/CUDA、有限梯度/实际更新、shape/部署证据沿用engineering-review.zh.md，候选fe_model.py未在服务器交付阶段修改。

本地环境为RTX4060 Laptop / torch2.9.1+cu128；服务器队列要求既有SG6的Python3.11.8 / torch2.2.2+cu121 / RTX4090环境并在服务器重新执行CUDA/入口合成检查，不把本地结果当作目标环境PASS。
本地首次delivery-check使用系统临时目录时遇到沙箱文件权限限制，已改用仓库tmp/下保留fixture；这属于工程路径限制，不是模型失败。

默认两组均从零50K。单组指标负向不代表训练进程失败；两组正常完成且备份和回传包校验通过后仍按授权关机。
真实分布、PSNR、目标机耗时/显存、数据路径、关机权限及平台最终关机结果均尚未验证；后台队列会在训练前或结束时分别检查对应条件。
没有自动安装依赖、升级环境、关闭TLS、删除旧结果、销毁实例或绕过源码干净检查。
