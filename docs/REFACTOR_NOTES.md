# 重构决策记录

## 旧版问题与新位置

| 旧版现象 | 重构后的约束 |
|---|---|
| 多个训练脚本各自维护参数和 PPO 循环 | `config.py`、`paper.yaml` 与单一 `PPOTrainer` |
| 策略或奖励依赖全局地图、BFS/测地距离、探索编码 | 策略仅接收 85 维局部观测，奖励只使用欧氏目标进展和局部事件 |
| D 分支拼接当前/历史特征，却没有显式差分 | `PIDMCell` 使用 `feature - previous_feature` |
| 循环状态在序列每一步 detach | rollout 采样后才 detach，训练序列内部保持计算图 |
| twin critic 取最小值后只训练一条价值分支 | 最小值仅构造 GAE；`clipped_twin_value_loss` 分别训练两条分支 |
| L 标签来自静态 BFS 最短路径 | 按 episode 结果动态生成，失败仅监督末尾 K 步 |
| 消融通过复制并手改大脚本完成 | `experiment.py` 的受控 variant 覆盖 |
| 测试地图重新训练，无法衡量泛化 | `evaluate.py` 只加载固定 checkpoint 跨地图评估 |

## 模块边界

环境负责物理状态、碰撞、LiDAR 和 reward；模型只处理张量；buffer 负责 trajectory 标签和序列切分；trainer 负责采样与优化。环境中的地图是仿真世界的一部分，不等于策略可见信息。

## 后续补充意见的落点

- 若审稿意见要求五随机种子统计，应新增聚合脚本读取每个 run 的结构化指标，不应修改算法代码。
- 若要求真实噪声或时延实验，应扩展环境 wrapper，不应改变 PIDM 或 PPO 损失。
- 若要求 PyBullet 复现，应新增与 `MaplessNavigationEnv` 同契约的环境类并增加后端一致性测试。
- 若新增对照算法（如 SAC/DDPG），应复用环境和评估协议，但使用独立算法模块，不能把它们伪装成 PPO variant。
