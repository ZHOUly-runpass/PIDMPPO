# 训练前实验矩阵审计

审计结论：三个训练前闸门均已通过，项目状态为 `ready_for_full_training`。核心训练矩阵、五种子协议、固定地图评测、主要消融、敏感性配置、机制轨迹和随机地图评测均已落地。本结论只表示运行链路完整，不代表短程模型已有有效性能。

审计依据为《ISA Transactions 大修意见分类汇总与返修分析》中的实验补充、统计协议、泛化验证和可复现性要求。此次审计不启动训练，也不产生论文结果。

## 审计后的规模

| 阶段 | 设置数 | Seeds | 训练任务 | 单任务步数 | 总步数 |
|---|---:|---:|---:|---:|---:|
| 核心矩阵 | 13 | 5 | 65 | 1,500,000 | 97,500,000 |
| 敏感性矩阵 | 12 | 5 | 60 | 1,500,000 | 90,000,000 |
| 合计 | 25 | 5 | 125 | - | 187,500,000 |

每个 checkpoint 在 Maps 1–4 上分别评测 100 episodes，共 400 episodes。每张地图保存 25 个固定 start-goal，评测时循环四次，所有方法使用同一列表。仅核心矩阵完整评测即产生 26,000 episodes。

## 大修意见与矩阵覆盖

| 审稿要求 | 对应设置或工具 | 状态 | 训练前判断 |
|---|---|---|---|
| Full PIDM-PPO | `pidmppo` | 已就绪 | 核心参照组 |
| 原始 LSTM/GRU 基线 | `lstm_ppo`, `gru_ppo` | 已就绪 | GRU 已补入训练矩阵 |
| LSTM auxiliary-only | `lstm_auxiliary` | 已补齐 | 用于拆分 auxiliary contribution |
| LSTM dual-value-only | `lstm_dual_value` | 已补齐 | 用于拆分 value baseline contribution |
| LSTM/GRU 公平匹配基线 | `matched_lstm`, `matched_gru` | 已就绪 | 与 PIDM 使用相同 auxiliary heads 和 dual value heads |
| D-like 消融 | `without_d` | 已就绪 | D 分支置零并有单元测试 |
| Attention 消融 | `without_attention` | 已就绪 | 使用均值融合替代可学习 attention |
| L/G 独立贡献 | `l_loss_only`, `g_loss_only`, `without_auxiliary` | 已就绪 | Full 作为 L+G 联合设置 |
| Single vs dual value | `single_critic`, `pidmppo` | 已就绪 | 轨迹含 V1、V2、min 和 MC return |
| 内部机制可视化 | `trace.py`, `plot_trace.py` | 部分就绪 | 能记录 memory、gate、P/I/D、动作和值；尚需选定成功/失败 trap episodes |
| Value error 与 bias | `analyze_values.py` | 已就绪 | 按 method 和 normal/trap-stagnation 分组，输出 MAE、bias、head correlation |
| Maps 2–4 泛化 | 固定 `scenarios.json` | 已就绪 | 每图 25 个固定场景，100 episodes/map/checkpoint |
| 随机地图与 trap geometry | `generate_random_maps.py` | 部分就绪 | 生成器可用，尚未加入统一评测调度 |
| 五种子与置信区间 | `aggregate_results.py` | 已就绪 | mean/std、median/IQR、bootstrap CI、success/collision/timeout Wilson CI |
| checkpoint 规则 | `final.pt` | 已就绪 | 禁止按评测集挑选最佳 checkpoint |
| 参数与资产来源 | manifest 和 provenance 文档 | 已就绪 | 重建地图、URDF 和停滞参数必须作为推定项报告 |
| 实机增强 | 不在本轮范围 | 暂缓 | 当前任务明确只完成补充仿真 |

## 核心训练矩阵

核心矩阵每项使用 seeds 11、22、33、44、55。

| 组别 | Variant | 主要回答的问题 |
|---|---|---|
| Full | `pidmppo` | 完整方法表现 |
| Baseline | `lstm_ppo` | 原始循环记忆基线 |
| Baseline | `gru_ppo` | 原始 GRU 记忆基线 |
| Factorial control | `lstm_auxiliary` | auxiliary supervision 单独带来的变化 |
| Factorial control | `lstm_dual_value` | dual value heads 单独带来的变化 |
| Matched baseline | `matched_lstm` | 公平比较 PIDM 与 LSTM encoder |
| Matched baseline | `matched_gru` | 公平比较 PIDM 与 GRU encoder |
| PIDM ablation | `without_d` | D-like 分支贡献 |
| PIDM ablation | `without_attention` | 可学习 attention fusion 贡献 |
| Auxiliary ablation | `without_auxiliary` | 总 auxiliary supervision 贡献 |
| Auxiliary ablation | `l_loss_only` | L-Loss 独立贡献 |
| Auxiliary ablation | `g_loss_only` | G-Loss 独立贡献 |
| Value mechanism | `single_critic` | conservative minimum baseline 的误差与偏差 |

`ppo`、`without_i` 仍保留为可选诊断 variant，但不进入本轮必要核心矩阵，避免扩大正式计算量而偏离审稿问题。

## 敏感性矩阵

论文返修的主要敏感性证据采用 memory bound、gate temperature 和 auxiliary weight，各自 low/high，共 6 个设置。停滞 window、epsilon、penalty 各自 low/high，共 6 个设置，用来验证未公开推定参数不会主导结论。每项仍使用五个独立训练 seeds。

建议正文优先报告前三类结构/学习超参数，将停滞参数结果放入补充材料或 robustness appendix。

## 评测与统计口径

- 训练和评测种子分离；训练种子固定为 11、22、33、44、55。
- checkpoint 固定使用 `final.pt`，不根据评测结果选择。
- 每个 checkpoint 在每张地图运行 100 episodes。
- 所有方法共享相同 `scenarios.json`，不能临时随机抽取不同 start-goal。
- success、collision、timeout 报告比例和 Wilson 95% CI。
- path length、elapsed time、steps、episode return 报告 mean/std、median/IQR 和 seed-level bootstrap 95% CI。
- 方法比较以训练 seed 的汇总值为抽样单位，不能把同一模型的大量 episode 当作独立训练重复。
- Map 2 必须单独报告，不能只给四图平均值。
- Value 机制报告 V1、V2、min 相对 MC return 的 MAE、E[V-G] bias 和两 head Pearson correlation，并分 normal 与 trap/stagnation 状态。

## 正式训练闸门验证结果

1. 13 个核心设置均完成 seed 11、4,096 environment steps；每项完成 4 次 rollout/update。
2. 13 个 final checkpoint 均可加载，训练 CSV、manifest 和 checkpoint 齐全；所有指标和模型张量均为有限值。
3. 固定 U-shaped trap 已建立，smoke checkpoint 已运行 5 个机制回合并生成 3,000 个时序样本、失败轨迹图和 value 分析。
4. 成功轨迹选择逻辑已经实现；短程模型没有成功逃逸，因此成功样本需在正式 checkpoint 产生后提取，不能伪造。
5. 60 个固定随机场景已生成，覆盖 easy/medium/hard、none/U/C/concave 和 5 个生成 seeds。
6. zero-shot 调度已用 smoke checkpoint 实际运行 60 episodes，并通过协议审计和按 difficulty 汇总。

正式训练启动前仍应冻结代码版本和环境版本；所有 variant 必须使用同一冻结版本。

## 推荐执行顺序

1. 运行矩阵审计：`python scripts/audit_experiment_matrix.py`。
2. 查看冒烟审计：`python scripts/audit_smoke_runs.py`。
3. 冻结代码、配置和软件环境版本。
4. 运行 65 个核心任务并完成固定 Maps 1–4 评测。
5. 核心结果完整且统计协议通过后，再运行 60 个敏感性任务。

## 最终判定

当前状态为 `ready_for_full_training`。仍需注意：Maps 1–4、URDF 和停滞参数属于明确记录的重建或推定项；smoke 结果只验证运行稳定性，不能写入论文作为性能证据。
