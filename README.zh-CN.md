# CouponIM-experience 使用说明

这是从论文实验工程中拆出的独立代码目录，包含核心算法、Python/C++ 实现、
基线、独立评估和实验调度。没有具体数据集、画图脚本、历史实验结果或编译产物。

## 最小运行流程

在本目录下执行（Python 3.10 及以上）：

```sh
python -m pip install -r requirements.txt
python -m couponim demo --output results/demo-python
```

示例在运行时生成 24 个节点的小图，选种、评估并导出 `raw.csv`、`summary.csv`、
种子列表和日志。示例只检查流程能运行，不用于验证论文的性能结论。

需要加速时，先安装支持 C++17 的 GCC/Clang，使 `g++` 或 `clang++` 可从命令行调用：

```sh
python -m couponim demo --backend cpp --output results/demo-cpp
```

程序会自动编译。Windows 推荐 MinGW-w64；也可以通过 `CXX` 指定编译器。

## 用自己的图跑实验

复制 `examples/external-graph.template.json`，修改图文件路径、节点预算、采样倍率、
评估次数和方法列表。路径相对于配置文件；边表使用从 0 开始的整数节点编号。
`directed` 必须明确设置。若要保留编号末尾的孤立节点，设置 `n`。
大图优先使用包含 `indptr`、`indices` 数组的 CSR NPZ。

```sh
python -m couponim run --config examples/external-graph.template.json --backend cpp --workers 2 --output results/my-study
```

`--workers 1` 串行，`--workers 2` 最多两个任务并行。同一条命令再次执行会跳过
已完成任务；如果选种已完成但评估中断，会复用选种检查点。Ctrl+C 或输出目录下
的 `STOP` 文件可以停止任务，继续前需要删除 `STOP`。没有在第 j 轮内部恢复的功能，
未完成的选种需重做；已保存的完整种子集会保留。

`balanced-reference` 是原先较低方差的参数，`balanced` 是保持均值、将消费概率
偏差放大三倍的参数。二者不会混用，所有输出都记录场景名称。算法保留当前实验代码
的行为：每轮仍对所有行采样，只将未覆盖行、未选节点的成员关系存入当前索引。

## 输出

- `raw.csv`：每个独立实验的实测评估值、用时。
- `summary.csv`：按数据集、环境、k、方法、采样倍率汇总；单次实验不填运行间标准差。
- `jobs/*.json`：实际种子、选种与评估随机种子、采样计数、内存等详细信息。
- `protocol.json`：配置、代码和图文件哈希、软件环境。
- `execution/` 与 `logs/`：运行状态、进程级采样内存和错误信息。

独立评估同时报告激活人数 `sigma(S)` 和核销数 `U(S)`。选种用的训练覆盖估计值
单独记录，不代替独立评估结果。并行执行的耗时会受资源竞争影响；比较运行时间请
使用串行方式。Windows 的峰值私有提交内存与 RSS 是不同指标，输出中分开记录。

## 验证

```powershell
$env:COUPONIM_TEST_CPP = '1'
python -m unittest discover -s tests -v
```

不设置环境变量时运行 Python 测试。设置后还会编译 C++，逐项比较完整样本库、
Python 流式算法和 C++ 流式算法在相同随机流下的结果。

完整接口、复杂度、实验配置与结果字段见英文 README 和 `docs/`。

## 完整实验目录（本次扩充）

现在不只提供最小示例，统一入口支持 **15 种选种方法**。可以先运行完整的小图检查：

```sh
python -m couponim methods
python -m couponim run --config examples/all-methods.json --backend cpp --workers 2 --output results/all-methods
```

- `couponim/`：可直接运行的算法、原生加速、参数生成、评估、诊断和调度。
- `examples/`：预算、环境、异质性、参数网格、采样量、算法稳定性和规模实验配置。
- `legacy/experiments/`：原实验驱动、核心实现、验证和审计源码快照。
- `legacy/gzc-impl/`：较早的模拟、选种、概率分布和矩阵处理代码。
- `docs/CATALOG.md`：所有功能、命令与原脚本之间的对应关系。
- `docs/source-inventory.json`：103 份原源码的迁入／排除清单及哈希。

**新实验使用 `couponim` 入口。** `legacy/` 中的 71 份文件是参考归档，保留原目录依赖，
并不承诺直接执行；早期模型与当前模型的差别也在归档说明中列出。归档没有进入安装包。

新增的条件采样、自适应采样、IC-RIS 等选种仍使用 Python。指定 `--backend cpp`
会对已有原生实现的采样和评估加速，结果中明确记录实际使用的后端。
MC-Greedy 和精确诊断有显式规模限制；自适应停止是经验规则，不代表理论证书。

以下命令分别用于采样器比较、小图精确检查，以及只重评估已保存的种子：

```sh
python -m couponim sampler-ablation --config examples/sampler-ablation.json --output results/ablation
python -m couponim diagnose --config examples/exact-diagnosis.json --output results/exact
python -m couponim reevaluate --source results/all-methods --output results/evaluation-5000 --trials 5000 --backend cpp --workers 2
```

详细配置说明见 [扩充参数说明](docs/EXTENDED_CONFIG.md)。所有示例图都在运行时生成；
没有迁入具体数据集、画图程序、预测／手工修改结果或历史结果文件。
