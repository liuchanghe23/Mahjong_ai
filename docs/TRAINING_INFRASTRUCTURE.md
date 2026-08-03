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

- `train`：600个配对种子，用于60、300、600个种子逐级筛选；
- `selection`：1002个配对种子，用于候选选择；
- `validation`：3000个配对种子，只用于最终验证。

所有种子数必须为6的倍数，从而完整覆盖六种2v2席位组合。每个种子运行两局A/B席位镜像对战，因此实际模拟局数为预算的两倍。加载器会拒绝重叠种子段和不平衡场数。

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

## 并行随机搜索

`scripts/train.py`实现确定性随机采样和Successive Halving。默认生成24个候选，使用7个进程并行评测，在60、300、600个配对种子后保留前25%。候选始终包含手工初始参数，并按平均顺位改善、点差改善、候选哈希依次排序。

```powershell
.\.venv\Scripts\python scripts\train.py
```

每个候选配置、每阶段JSON/Markdown报告、阶段排名、运行清单和最终摘要都会立即写入`artifacts/training/stage1/`。相同输出目录再次运行时读取manifest并跳过已经完成的候选预算，可用于中断后继续。

正式训练默认要求Git工作区没有未提交修改。运行清单会记录代码提交、训练规格指纹和全部关键参数；续跑发现其中任何一项变化都会停止，避免混用不同版本的结果。开发中的一次性冒烟测试可加`--allow-dirty`。

8核CPU默认使用7个工作进程。内存不足时可降低并行度：

```powershell
.\.venv\Scripts\python scripts\train.py --workers 4
```

快速验证进程与输出流程可使用：

```powershell
.\.venv\Scripts\python scripts\train.py `
  --candidates 2 --budgets 6 --workers 2 `
  --bootstrap-samples 200 `
  --output-dir artifacts/training/smoke `
  --allow-dirty
```

## 三参数单变量消融

`configs/training/group-weight-ablation.yaml`定义13个确定性候选：`1/1/1`基线，以及结构、价值、风险各自4个非基线水平。每个非基线候选只改变一个参数，统一使用300个镜像配对种子，不进行早期淘汰。

```powershell
.\.venv\Scripts\python scripts\ablate_group_weights.py
```

默认使用7个工作进程，输出到`artifacts/training/group-weight-ablation-v1/`。其中`summary.md`按参数给出趋势表，`summary.json`保留完整机器可读结果。重复运行同一命令会跳过已经完成的候选；正式运行同样要求Git工作区干净。
