# the-first-paper · 固态电解质 AIMD/MD 轨迹科研代码

第一版目标：在 MacBook Air M1 上跑通 **轨迹 → 统一表示 → 时间切分 → 简单基线 → MSD/实验记录**。
核心流程仅使用 Python 标准库，支持 Python 3.9+；Transformer 和 ASE 为可选依赖。

**示例是相关随机游走合成数据，不是 Li₃OCl、LLZO 或其他真实材料的模拟结果。**
仓库提供轨迹处理和时序预测起点，尚不构成经过验证的物理模型或论文结论。

## 立即运行（MacBook Air M1）

在仓库目录打开终端：

```bash
python3 --version
python3 scripts/run_demo.py
python3 -m unittest discover -s tests -v
```

无需 pip、NumPy、GPU、LAMMPS 或 VASP。脚本也可以从其他目录通过绝对路径调用。
首次运行生成 `data/synthetic/demo.json` 和 `experiments/runs/demo/`。
重复运行使用 `python3 scripts/run_demo.py --force`；该选项会覆盖该示例的数据与结果。

| 输出 | 内容 |
| --- | --- |
| `report.json` | 配置、来源哈希、切分边界、验证/测试误差、MSD 拟合与运行环境 |
| `ridge_model.json` | 岭回归权重及训练集均值/尺度 |
| `msd.csv` | 时间原点平均的 Li MSD，时间单位 ps、MSD 单位 Å² |
| `test_predictions.csv` | 测试集真实/预测下一步位移，单位 Å |

比较零位移、上一帧位移延续、岭回归三种基线；不预设复杂模型一定更好。
默认 480 帧、6 个原子、4 步历史；先切帧再建窗口，三个集合不共享原始帧。
该单轨迹时间留出只用于流程检查，正式研究需要独立轨迹/体系分组验证。

## 目录

| 路径 | 用途 |
| --- | --- |
| `src/solidmd/trajectory.py` | JSON 数据契约、校验、PBC 解包裹、合成示例 |
| `src/solidmd/adapters.py` | LAMMPS dump、普通 XYZ、可选 ASE 读取 |
| `src/solidmd/features.py` | Li 位移历史、时间切分、误差指标 |
| `src/solidmd/models.py` | 标准库实现的小规模岭回归 |
| `src/solidmd/analysis.py` | MSD 和指定区间扩散拟合 |
| `src/solidmd/pipeline.py` | 流程编排和可追溯实验记录 |
| `scripts/` | 开箱即用的演示、真实数据导入、训练入口 |
| `configs/` | 演示配置和本地路径模板 |
| `data/raw/` | 原始数据约定；真实轨迹不入 Git |
| `data/processed/`、`data/synthetic/` | 统一轨迹和明确标注的合成数据 |
| `experiments/` | 实验约定和忽略提交的运行产物 |
| `tests/`、`.github/workflows/` | 物理/数据切分/模型测试和 GitHub CI |
| `docs/` | 数据接口、限制和接入路线 |

## 接入四种引擎

| 来源 | 第一版入口 | 需要你补充 |
| --- | --- | --- |
| LAMMPS | 原生 `lammps-dump`，无需依赖 | type→元素映射；Å 坐标；固定正交胞；优先 `xu yu zu` |
| DFTB+ | 普通 XYZ，`--format xyz` | 固定盒长、输出帧间隔、稳定原子顺序 |
| VASP | ASE 的 `vasp-xdatcar` | 安装 `.[io]`；固定正交胞 XDATCAR；输出帧间隔 |
| CP2K | 普通 XYZ，`--format xyz` | 固定盒长、Å 输出、输出帧间隔、稳定原子顺序 |

例如导入 LAMMPS 未包裹轨迹（将路径、映射和帧间隔替换为真实设置）：

```bash
python3 scripts/convert_trajectory.py data/raw/run.dump data/processed/run.json \
  --engine lammps --format lammps-dump \
  --timestep-fs 20 --coordinate-mode unwrapped --trajectory-id run-001 \
  --type-map '{"1":"Li","2":"O","3":"Cl"}'
```

导入 CP2K XYZ（以下 10 Å 盒长和 2 fs 仅是命令参数示例）：

```bash
python3 scripts/convert_trajectory.py data/raw/run.xyz data/processed/run.json \
  --engine cp2k --format xyz --box 10 10 10 \
  --timestep-fs 2 --coordinate-mode unwrapped --trajectory-id run-001
```

随后复制 `configs/local.example.json` 为 `configs/local.json`，填写 `input` 和独立的 `output`。
配置中的相对路径以**仓库根目录**为基准，也支持外置硬盘的绝对路径。
按轨迹长度调整 `msd_max_lag` 和拟合区间，再运行：

```bash
python3 scripts/run_pipeline.py --config configs/local.json
```

普通 XYZ 缺乏晶胞和时间信息，必须自行确认；ASE 不会自动解决单位/原子身份问题。
`timestep_fs` 指相邻保存帧的时间间隔。包裹坐标要求每帧每轴真实位移小于半盒长。
不支持变胞/斜晶胞，输入不满足时会报错。大轨迹请先使用连续小片段验证。
详见 [数据契约](docs/DATA_CONTRACT.md) 和 [研究接入顺序](docs/ROADMAP.md)。

## 可选：Transformer 和 ASE

建议用 Python 3.11/3.12 独立虚拟环境安装可选依赖；标准库演示可继续使用系统 Python 3.9。
如果没有 `python3.11`，先安装对应 arm64 Python，再执行以下命令：

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[io,transformer]'
python scripts/run_demo.py --force
python scripts/train_transformer.py --epochs 5 --device cpu
```

Transformer 采用 1 层、24 维、2 个注意力头，以 Li 的历史位移预测下一步位移。
根据验证集选择 epoch，保存权重、训练尺度、配置和测试误差。
M1 可选 `--device mps`，不可用时请用 CPU；CPU 是默认值。
它尚无局域环境或对称性编码，不能用于可靠的长期轨迹生成。

VASP 导入示例（安装 ASE 后）：

```bash
python scripts/convert_trajectory.py data/raw/XDATCAR data/processed/vasp.json \
  --engine vasp --format vasp-xdatcar --timestep-fs 2 \
  --coordinate-mode wrapped --trajectory-id vasp-001
```

## GitHub 与研究记录

仓库忽略原始/处理数据、大模型权重和运行结果；不要上传 VASP POTCAR、私密轨迹或未授权数据。
数据放外部盘或合适的数据存储平台，仓库记录路径、来源和哈希。
`report.json` 保存 Git commit 和 dirty 状态；按 [实验约定](experiments/README.md) 使用独立运行目录。
连接 GitHub 后提交源码即可触发 macOS/Linux、Python 3.9/3.12 的核心 CI。
CI 配置已提供，远端运行需要推送后确认。

如果当前文件夹还没有 Git 元数据，在自己的终端执行：

```bash
git init -b main
git add .
git commit -m "Initial trajectory pipeline and runnable demo"
```

然后在 GitHub 建立空仓库，按 GitHub 给出的命令添加 remote 并推送。
本次只建立本地源码和运行结果；没有创建或推送远端仓库。

当前未指定开源许可证；公开前由作者决定代码许可和数据可用范围。
