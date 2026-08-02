# 基线模型消融实验

消融实验让完整基线与“只关闭一组特征”的版本在同一牌山、同一场对局中直接竞争，用于判断该特征是否真的提升结果。

## 实验配置

`configs/ablations/` 中的配置继承 `configs/baseline.yaml`，只覆盖指定权重：

- `no-risk`：关闭弃牌危险度。
- `no-yaku`：关闭四类役潜力增量。
- `no-all-yaku`：在 `no-yaku` 基础上关闭役牌对子形状奖励。
- `no-dora`：关闭宝牌与赤宝牌保留奖励。
- `no-lookahead`：关闭一巡Expectimax有效牌质量特征。
- `no-shape`：关闭两面、嵌张、边张、对子与役牌对子形状奖励。
- 历史报告中的`no-isolated`已停止使用：旧孤张扫描已被结构分解中的四类`unused_*`惩罚替代。

覆盖式配置避免复制完整参数；以后修改基线时，各消融版本仍只与基线相差目标权重。

## 运行方法

筛选实验使用东风战、600 场（场数应为 6 的倍数，以完整覆盖六种席位组合）：

```powershell
.venv\Scripts\python scripts\ablate.py --matches 600 --base-seed 30000
```

确认实验建议在开发种子和独立验证种子上分别运行 3000 场：

```powershell
.venv\Scripts\python scripts\ablate.py --matches 3000 --base-seed 40000
.venv\Scripts\python scripts\ablate.py --matches 3000 --base-seed 80000
```

可用 `--ablations no-risk no-yaku` 只运行指定实验。

## 统计解释

JSON 报告保存每场种子、完整模型席位和双方团队指标，便于复查。报告还按“对局”进行 10,000 次配对 Bootstrap，给出 95% 置信区间。平均顺位、四位率、点差、和牌率和放铳率均统一定义为正数代表完整模型更优。

平均顺位、平均点差、四位率是主要指标；和牌率、放铳率用于解释原因。若区间跨过 0，当前样本尚不足以确认该特征有效。多个消融同时下结论时，应额外使用 Holm 方法校正显著性。

正式实验使用东风战而非单局模式，以降低单局偶然性以及未领取立直棒对点差的干扰。
