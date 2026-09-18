# 大修意见代码实现矩阵

本表记录代码能力，不代表实验结论。详细的训练任务、统计口径和启动闸门见 `TRAINING_READINESS_AUDIT.md`。

| 大修任务 | 代码位置 | 当前能力 | 仍需完成 |
|---|---|---|---|
| 公平 matched baseline | `experiment.py` | LSTM/GRU 可分别使用原始、aux-only、dual-only 和完整匹配设置 | 正式五种子结果 |
| D 与 attention 消融 | `experiment.py`, `models/pidm.py` | D 可关闭，attention 可替换为均值融合 | 正式结果 |
| L/G auxiliary 消融 | `experiment.py`, `algorithms/auxiliary.py` | L-only、G-only、无 auxiliary 与 Full | 正式结果 |
| PIDM 机制轨迹 | `evaluation/trace.py`, `plot_trace.py` | memory、gate、P/I/D attention、动作、轨迹和值 | 选定成功/失败 trap episodes |
| Dual value 机制 | `analyze_values.py` | MAE、bias、minimum error、head correlation，按方法和状态分组 | Single/Dual checkpoint 配对结果 |
| 五种子统计 | `experiment_matrix.yaml`, `aggregate_results.py` | 65 个核心任务；Wilson、bootstrap、mean/std、median/IQR | 计算资源与正式运行 |
| 固定评测协议 | `scenarios.json`, `evaluate.py` | Maps 1–4 每图 25 个固定 start-goal，100 episodes/map/checkpoint | 正式 checkpoint |
| 随机地图泛化 | `generate_random_maps.py` | 密度分级、U/C/concave trap、固定生成 seed | 接入批量评测调度 |
| PyBullet 补充仿真 | `envs/backends.py`, `paper_pybullet.yaml` | 内置 URDF、wheel control、ray-test LiDAR、线段地图 | 训练前短程运行验证 |
| 可复现清单 | `utils/manifest.py`, `PARAMETER_PROVENANCE.md` | 保存参数、软件版本、seed、git commit 与推定项 | 正式环境冻结 |
| 实机统计 | `summarize_real_robot.py` | success/collision/timeout/path/time 统计 | 本轮补充仿真不执行实机试验 |

Maps 1–4、URDF、摩擦参数和停滞参数是可复现重建或推定值，不得在返修稿中表述为找回的原始实验资产。
