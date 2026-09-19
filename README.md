# PIDM-PPO 返修实验代码

本项目依据论文 *PID Inspired Gated Residual Memory for Mapless Mobile Robot Navigation with Dual Value Heads* 重构训练、消融、评测和补充仿真代码。目前仅完成短程冒烟训练，尚未执行五种子正式训练，不产生论文性能结论。

项目仓库：https://github.com/ZHOUly-runpass/PIDMPPO

## 当前实现

- 85 维观测：64 路 LiDAR、3 维相对目标、最近 9 个二维执行速度。
- PIDM 的 P-like、bounded I-like、latent-difference D-like 和 attention fusion。
- 保留序列 BPTT 的 recurrent PPO、双 value head、L/G auxiliary loss。
- PIDM、LSTM、GRU、公平因子对照、单 critic、去 D、去 attention、去 auxiliary 等 15 个统一变体。
- 论文 Figure 5 约束下重建的 Maps 1–4 米制线段几何。
- 内置 TurtleBot3 Burger 仿真近似 URDF，轮关节索引固定为 left=0、right=1。
- 栅格后端和可选 PyBullet 差速运动学、ray-test LiDAR、碰撞检测。
- 固定评测场景、五种子任务矩阵、随机陷阱地图、统计与轨迹诊断。

代码完整不等于原始实验已恢复。地图尺度、URDF 动力学和停滞参数属于明确标注的重建/推定项，详见 [参数来源](configs/PARAMETER_PROVENANCE.md)。

## 安装与无训练验证

```powershell
python -m pip install -e .
python -m pytest
python scripts/validate_protocol.py
python scripts/model_report.py
python scripts/inspect_simulation_assets.py
python scripts/audit_experiment_matrix.py
```

需要 PyBullet 或地图预览时：

```powershell
python -m pip install -e ".[simulation,analysis]"
python scripts/inspect_simulation_assets.py --render artifacts/map_previews
```

`inspect_simulation_assets.py` 检查地图起终点、米制尺寸、墙体资源、URDF 碰撞几何和关节声明顺序，不会启动训练。

## 仿真配置

- `configs/paper.yaml`：快速测试和算法开发使用的栅格后端。
- `configs/paper_pybullet.yaml`：补充仿真使用的 PyBullet 后端和内置 URDF。
- `src/pidmppo/assets/maps/paper_maps.json`：Maps 1–4 数值几何。
- `src/pidmppo/assets/robots/turtlebot3_burger_sim.urdf`：仿真机器人。

PyBullet 配置核心字段：

```yaml
env:
  backend: pybullet
  robot_urdf: bundled:turtlebot3_burger_sim.urdf
  left_wheel_joints: [0]
  right_wheel_joints: [1]
  wheel_radius: 0.033
  axle_length: 0.160
```

地图是根据论文图示拓扑与相对比例建立的可复现版本，并非声称找回了原始数值资产。内置 URDF 只服务补充仿真，不代表实车质量、惯量、摩擦或传感器标定。

## 停滞奖励基线

论文给出了弱停滞惩罚形式，但没有公开三个数值。当前基线为：

- `Hs = 20 steps`，对应 2.0 s；
- `epsilon = 0.001 m/step`，即 2 s 总进展不足约 0.02 m；
- `penalty = -0.02`。

`configs/experiment_matrix.yaml` 同时包含窗口、阈值和惩罚的低/高敏感性配置。正式补充实验应完整报告这些组合，不能将基线写成论文原始参数。

## 任务矩阵（默认只打印）

```powershell
python scripts/run_matrix.py --stage core
python scripts/run_matrix.py --stage sensitivity
python scripts/evaluate_matrix.py
```

正式矩阵为 13 个核心训练设置 × 5 seeds = 65 个任务，另有 12 个敏感性设置 × 5 seeds = 60 个任务。首次运行前应先使用 `configs/smoke_matrix.yaml` 做单种子短程验证。

只有显式增加 `--execute` 才会启动训练或评测；通用训练矩阵还必须明确 `--max-jobs`。运行前应先确认输出目录、计算预算和冻结版本，不使用暂停父进程控制队列。

机制场景与随机地图评测：

```powershell
python scripts/evaluate_mechanism_matrix.py
python scripts/generate_random_maps.py --output generated_maps
python scripts/evaluate_generalization_matrix.py
```

以上命令默认只打印正式 checkpoint 任务；增加 `--execute` 才会执行。机制评测完成后，使用 `select_mechanism_episodes.py` 自动选取成功与失败轨迹并生成对比图。

## 固定评测协议

旧 `configs/scenarios.json` 的 25/图场景仅保留用于追溯，不再作为新训练正式评测集。

当前采用 `configs/scenarios_v2/`：固定地图每图 100 个独立任务、训练分布验证 100 个任务、随机 zero-shot 60 张地图，全部经过 0.15 m 净空、零动作和 ≤540-step 参考控制器验收。场景及几何有 SHA256，新评测禁止重复扩充样本。

## Correctness-v2 限时优化

动作概率、squashed entropy、超时 bootstrap、净空采样和完整训练诊断已接入。保留 PIDM、双价值头和 L/G；旧 checkpoint 保持可推理，新训练从头写入独立目录。

[实施协议与命令](docs/OPTIMIZATION_V2.md) 包含 A–D 筛选、三种子确认、24 小时/11 任务上限和自动 HOLD 条件。

```powershell
python scripts/run_preflight_v2.py
python scripts/run_optimization.py              # 只打印计划
python scripts/run_optimization.py --execute    # 预检通过后执行，绝不恢复原 65 任务
```

正式评测只使用固定 checkpoint，不挑选最佳 seed；`episodes.csv` 是汇总程序的唯一结果输入。

## 目录

```text
configs/                    参数、PyBullet 配置、任务矩阵和固定场景
scripts/                    验证、训练、评测、统计和绘图入口
src/pidmppo/assets/         地图 JSON 与仿真 URDF
src/pidmppo/algorithms/     PPO、GAE、recurrent buffer 和 auxiliary loss
src/pidmppo/envs/           栅格/PyBullet 后端与地图生成
src/pidmppo/evaluation/     场景协议、轨迹记录和统计
src/pidmppo/models/         PIDM、MLP、GRU、LSTM 和 actor-critic
tests/                      数学约束、资源和接口测试
```
