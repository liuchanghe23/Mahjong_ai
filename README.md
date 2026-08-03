# Mahjong AI

立直麻将 AI 的本地研究与复盘项目。第一阶段使用 RiichiEnv 建立可重复的牌局模拟闭环，后续再接入 MJAI 决策引擎、解释器和截图识别。

## 当前结构

- `src/mahjong_ai/`：本项目代码
- `scripts/`：本地运行入口
- `tests/`：自动化测试
- `vendor/RiichiEnv/`：上游 RiichiEnv 源码（独立 Git 仓库，不纳入本项目版本控制）

## 本地启动

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -e .
.\.venv\Scripts\python scripts\smoke_simulator.py
```

成功时会运行一局固定种子的四人麻将随机对局，打印结果，并将通过解析校验的 MJAI 日志写入 `artifacts/replays/smoke.jsonl`。

## Mortal 接入边界

`DecisionEngine.act(observation)` 是统一决策接口。运行器会在每一步检查引擎返回值是否属于 `observation.legal_actions()`；未来 Mortal 适配器只需消费 `observation.new_events()`，再用 `observation.select_action_from_mjai()` 将模型响应转换为合法动作。

`MjaiBotEngine` 已实现上述事件桥接。接入 Mortal 时，只需提供带有 `react(event)` 方法的状态化 Bot 实例，并在 `EngineFactory` 中按座位创建实例。

当前已具备：

- 雀魂规则预设
- 固定环境种子和独立玩家种子
- 可重复的批量模拟
- MJAI JSONL 日志保存和解析校验
- 引擎非法动作拦截

## 可解释基线模型

`BaselineEngine` 已提供模块化的单步决策基线，权重位于 `configs/baseline.yaml`。运行：

```powershell
.\.venv\Scripts\python scripts\smoke_baseline.py
```

完整决策顺序、特征定义、评分方式和已知限制见 [`docs/BASELINE_MODEL.md`](docs/BASELINE_MODEL.md)。`calculate_shanten()` 的结构覆盖范围、非役种职责和常见误区见 [`docs/CALCULATE_SHANTEN.md`](docs/CALCULATE_SHANTEN.md)。

动作前后的役牌、断幺九、七对子和染手潜力判断见 [`docs/YAKU_VALUE_MODEL.md`](docs/YAKU_VALUE_MODEL.md)。

硬向听优先级、特征归一化、模块权重和风险上下文训练接口见 [`docs/TRAINABLE_SCORING.md`](docs/TRAINABLE_SCORING.md)。

训练基线、参数约束、种子隔离和候选配置快照见 [`docs/TRAINING_INFRASTRUCTURE.md`](docs/TRAINING_INFRASTRUCTURE.md)。

固定种子评测、2对2座位轮换、指标口径和报告格式见 [`docs/EVALUATION_FRAMEWORK.md`](docs/EVALUATION_FRAMEWORK.md)。

消融配置、东风战实验流程与配对 Bootstrap 统计见 [`docs/ABLATION_EXPERIMENTS.md`](docs/ABLATION_EXPERIMENTS.md)。

只使用向听数和有效牌的合理对照模型见 [`docs/EFFICIENCY_CONTROL.md`](docs/EFFICIENCY_CONTROL.md)。

## 设计边界

项目面向本地模拟、截图分析和牌谱复盘，不包含游戏进程注入、网络拦截、自动点击或真人匹配中的自动操作。
