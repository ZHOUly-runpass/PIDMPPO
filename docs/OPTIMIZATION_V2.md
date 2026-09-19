# PIDM-PPO correctness-v2：实施协议与验收

本轮只进行 grid 后端的实现修复、验证及有条件优化，不宣称复现原稿 PyBullet 实验。旧五种子、checkpoint、旧评测和 `configs/scenarios.json` 保留。

## 实现修复

- rollout 保存动作和 `raw_action`；更新使用原始高斯样本及稳定 log-Jacobian，不再反推饱和动作。熵奖励使用当前策略新采样的 squashed entropy，高斯熵仅作诊断。
- 超时用 reset 前最终观测、处理当前观测后的循环状态 bootstrap；真正终止不 bootstrap。两类结束都截断 GAE，padding 不参与损失。遵循 [Gymnasium 时间限制语义](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/)。
- PIDM、双价值头最小值 GAE、L/G 定义、奖励和 600-step 时限保留。A 保留旧超参数。独立 value clip 参数默认回退旧值，尚未调参；其尺度问题见 [SB3 PPO 实现说明](https://stable-baselines3.readthedocs.io/en/master/_modules/stable_baselines3/ppo/ppo.html)。
- 起点、目标及路径统一按障碍实体矩形和地图边界保留 0.15 m 中心净空。Grid 圆形碰撞检测改为精确圆–矩形检测，补上旧四方向探针漏掉的角点。
- D 的难度变化只在 episode reset 应用；在 50k/100k 边界正在进行的 episode 会先结束。其余随机训练使用原完整距离分布并额外要求净空连通。

## 新资产

`configs/scenarios_v2/` 包含固定地图 400、训练分布验证 100、随机 zero-shot 60 个任务。固定地图每图 100 个不同的有序起终点，不靠改变 ID、朝向或重复记录扩充任务数。随机地图保留原 3×4×5 几何与种子，只重新选择 2–8 m 参考路径的有效任务。

验证资产来自 Map 1 同一随机走廊生成器的独立种子 9100000–9100099。任务保持完整训练距离下限，并通过参考控制器验收。参考控制器可见地图，只用于验收，绝不进入策略观测或提供训练动作。参考路径是净空栅格最短轴向路径的可见性简化，不声称是连续空间全局最短路径；成功路径效率为 `min(1, reference_path_length / executed_length)`。

每个 manifest 有 SHA256，外部地图另有文件哈希。内置地图另核验占据栅格哈希。v2 评测禁止超出独立任务数重复取模，支持每个验证任务独立地图文件；默认从 checkpoint 恢复配置，避免将 C/D 的 log-std 范围误恢复为 A。

初次生成的 560 个任务全部通过验收，最长参考控制器用时 371 steps。`audit_scenarios_v2.py` 会从文件独立重新运行全部验收，而非只检查生成器标签。

## 使用

```bash
python -m pytest -q
python scripts/audit_scenarios_v2.py
python scripts/run_preflight_v2.py
python scripts/run_optimization.py                         # 只打印预算
python scripts/run_optimization.py --execute               # 预检通过后限时执行
```

预检包括全部回归测试、560 场景独立复核、A–D 各 4096-step 完整 512 维网络冒烟训练，以及旧五种子 checkpoint 兼容性检查。优化调度没有通过预检就拒绝启动。所有输出目录必须为新目录，不会覆盖旧结果。

## 冻结筛选规则

`configs/optimization_v2.yaml` 定义 A–D，依次做旧五模型固定新场景复评、seed 11 每组 300k steps 筛选。只按训练分布验证成功率排序，平手依次比较碰撞率和成功路径效率；B/C/D 未比 A 提高 5 个百分点则停止扩大训练。

达到筛选线才从头训练 A 和最佳组 × seeds 11/22/33 × 500k，以及同样公共 PPO 设置的标准 `gru_ppo` 500k 诊断对照（GRU 对照按原定义为单价值头、无 L/G；不是参数匹配消融）。最大 11 个训练任务，名义 470 万步。rollout 取整后最大 **4,739,072** 步。另有预检 4×4096 步，不计入筛选性能。

24 小时预算从调度器启动起包含旧模型复评、训练和验证。距截止 15 分钟不再派发新任务；当前训练在截止前请求停止，在安全边界保存 `interrupted.pt` / `latest.pt`，不伪造 `final.pt`。预检是启动前工程验收，其耗时另记，若要求整个实施过程也包含在 24h 内，应使用 `--max-hours` 扣除已耗时间。停机、任务上限或中断状态均写入 `status.json`；不通过恢复命令重新计时、不自动恢复 65 项正式任务。

三种子验收要求：平均相对 A 提升 ≥10 个百分点，≥2 seeds 改善，最差 ≥10%，平均 ≥30%，平均碰撞率不增加。为使“明显减少旋转/停滞”可执行，预先定义为两项均不增加且两项之和下降 ≥10%（若基线为零则保持零）。不达标为 HOLD；达标也只标注“可进入人工复核”，不会自动启动正式矩阵。

## 诊断和结果

每个 update 保存 KL、裁剪率、实际轮数、裁剪前梯度范数、critic explained variance/值/return 分布、实际动作及潜在分布、P/I/D 权重、gate/memory、episode 结局及奖励分项。无已完成 episode 或零 return 方差时使用 0 和对应计数/有效性标记，不输出 NaN。完整跨 rollout episode 记录保存在 `training_episodes.jsonl`。

每 10 次 update 的第一个有效 minibatch 记录加权 actor（含熵项）、critic、auxiliary 对共享编码器的梯度范数；不将其解释为全 rollout 梯度。`health.json` 仅预警，不自动调低 value_coef。每次验证另外保存已有结局中各一条成功/碰撞/超时轨迹，未发生的结局不会编造。

`runs/optimization_v2/status.json`、`report.md` 持续更新，未填项就是未完成，不代表零分。三种子比较使用配对训练种子 bootstrap；n=3 的区间仅作探索性报告，不把 scenario 或 episode 重复当作独立训练样本。验证、固定 Map 1、未见 Maps 2–4 与随机 zero-shot 应分别报告，原稿主泛化仍对应 Maps 2–4。

新训练步数/参数等偏离原稿的设置仅作为返修优化阶段记录；奖励调参、PyBullet 验证和正式五种子公平对比仍是后续独立阶段。
