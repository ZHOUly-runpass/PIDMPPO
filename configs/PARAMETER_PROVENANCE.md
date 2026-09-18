# 论文参数来源与推定项

本清单区分论文明确给出的参数、由公式直接推导的实现，以及原始资产遗失后为补充仿真而建立的可复现推定。推定值不能在返修稿中写成原论文既定设置；正式结果必须同时报告敏感性分析。

## 论文明确给出

| 项目 | 数值或定义 |
|---|---|
| Policy observation | 64 LiDAR + 3 relative goal + 9 × 2 velocity history = 85 |
| LiDAR | 360°，裁剪至 0.12–3.5 m，并归一化至 0–1 |
| Linear velocity limit | 0.26 m/s |
| Angular velocity limit | 1.82 rad/s |
| Goal threshold | 0.20 m |
| Collision threshold | minimum LiDAR < 0.13 m |
| Control period | 0.1 s |
| Episode horizon | 600 |
| Terminal rewards | goal +100，collision -100 |
| Step penalty | -0.01 |
| Angular penalty | -0.01 × absolute angular velocity |
| PPO parameters | 见 `paper.yaml` 和论文 Table 2 |
| PIDM | Equations 13–22 和 Algorithm 1 |
| L target and mask | Equations 30–32 |
| G target | Equation 33，目标 PPO return 停止梯度 |
| Dual values | Equations 35–38 |

## 由公式或结构图确定的实现

- P module 使用 Linear → ReLU → Linear → Tanh。
- D module 输入当前特征与有限差分的拼接。
- Memory candidate 使用 Tanh，memory update 是门控凸组合。
- L-Loss 是 `0.5 × masked batch expectation`。
- 两个 value head 的最小值只用于 GAE/return 构造。
- 两个 value head 分别计算 clipped value loss。

## 为补充仿真建立的可复现推定

| 项目 | 当前定义 | 依据与处理 |
|---|---|---|
| Maps 1–4 | 6.5 × 4.0 m，0.10 m 栅格，0.06 m 墙厚 | 按论文 Figure 5 的拓扑和相对比例重建；原始数值几何不可得 |
| Map 1 obstacle randomization | 可复现 corridor-segment generator | 与论文“随机障碍/起终点”描述一致，但分布为新定义 |
| Simulation URDF | `turtlebot3_burger_sim.urdf` | 仅为仿真近似，不代表实车标定 |
| Wheel joints | left index 0，right index 1 | 由内置 URDF 的声明顺序固定，并在加载时校验名称 |
| Wheel radius / axle | 0.033 m / 0.160 m | TurtleBot3 Burger 尺度近似，用于差速运动学 |
| Robot collision radius | 0.105 m | 仿真近似；与论文 0.13 m LiDAR collision threshold 分离 |
| Stagnation window Hs | 20 steps = 2.0 s | 在 0.1 s control period 下提供短时停滞检测 |
| Stagnation epsilon | 0.001 m/step mean progress | 等价于 2 s 内总前进不足约 0.02 m |
| Stagnation penalty | -0.02 | 弱于终止奖励，和逐步/角速度惩罚同量级 |
| Friction / solver | 0.8 / 50 iterations | 明确记录的 PyBullet 数值设置，非论文原值 |
| Sensor noise / latency | disabled | 尚无可依据的噪声模型 |
| Goal-distance normalization | current map diagonal | 明确实现选择 |
| Velocity-history normalization | commands divided by limits | 明确实现选择 |
| Continuous policy | state-dependent diagonal Gaussian + Tanh | 明确实现选择 |

停滞窗口、阈值和惩罚均已纳入 `experiment_matrix.yaml` 的低/高敏感性组合。以上所有值会写入 `run_manifest.json`。
