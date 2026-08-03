# 参数训练基础设施

## 训练基线

`configs/training/baseline.yaml`继承当前v4基线，但将四项启发式役潜力与历史牌形兼容特征固定为0。向听硬优先、有效牌、Lookahead、不重叠结构、宝牌和风险保持启用。训练基线独立于稳定基线，搜索不会原地覆盖`configs/baseline.yaml`。

## 第一阶段搜索空间

`configs/training/stage1-search.yaml`只开放三个模块级参数：

- `group_weights.shape`：0～2；
- `group_weights.value`：0～2；
- `group_weights.risk`：0.5～3。

`group_weights.efficiency`固定为1，归一化尺度和全部细粒度权重冻结，避免模块权重与特征权重互相缩放。参数Schema严格校验路径、重复路径、有限数值、上下界、初始值和线性/对数尺度。未登记的配置路径不能被训练器修改。

## 种子隔离

`configs/training/seed-sets.yaml`定义互不重叠的三组种子：

- `train`：600场，用于60、300、600场逐级筛选；
- `selection`：1002场，用于候选选择；
- `validation`：3000场，只用于最终验证。

所有场数必须为6的倍数，从而完整覆盖六种2v2席位组合。加载器会拒绝重叠种子段和不平衡场数。

## 候选配置与复现

`apply_parameters()`只接受搜索空间中完整且合法的参数集合，返回内存中的不可变`BaselineConfig`。`BaselineEngine(config=...)`可以直接运行候选，无需覆盖磁盘基线。

每个候选可以生成：

- 与名称无关的SHA-256配置哈希；
- 可重新由严格加载器读取的完整YAML快照；
- 基于哈希的确定性候选名称。

运行以下命令验证训练配置并生成初始候选快照：

```powershell
.\.venv\Scripts\python scripts\validate_training.py
```

下一阶段将在此Schema上增加候选采样、分阶段淘汰、结果清单和断点恢复，不需要重新定义参数边界。
