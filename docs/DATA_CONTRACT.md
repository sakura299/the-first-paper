# 统一轨迹与导入约定

版本 1 的 JSON 含以下字段，完整示例由 `python3 scripts/run_demo.py` 生成。

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 整数 1 |
| `units` | `{"length":"angstrom","time":"fs"}` |
| `timestep_fs` | 相邻保存帧的时间间隔，有限正数 |
| `cell_angstrom` | 固定 3×3 对角晶胞矩阵，晶胞矢量按行存储 |
| `pbc` | `[true,true,true]` |
| `coordinate_mode` | `wrapped` 或 `unwrapped`，由使用者明确提供 |
| `species` | 长度 N 的元素列表，原子索引跨帧保持身份一致 |
| `positions_angstrom` | 形状 T×N×3 的笛卡尔坐标 |
| `metadata.engine` | 来源引擎 |
| `metadata.data_kind` | `simulation` 或 `synthetic` |
| `metadata.trajectory_id` | 独立模拟的 ID，不能把相同轨迹切片伪装为独立轨迹 |

第一版不支持变胞、斜晶胞、非均匀帧间隔、变原子数及能量/力标签。无晶胞的 XYZ 必须明确补充固定晶胞。
所有导入都要求输入已经采用 Å；LAMMPS `metal`/`real` 的长度可用，`lj` 等约化单位必须先转换。
所有读取器载入完整轨迹，JSON 用于小规模验证；大轨迹应先截取平衡后的连续小片段，后续升级流式 HDF5/Zarr。

## 原子身份与周期边界

LAMMPS 按 `id` 排序并校验每帧 ID/类型，支持 `xu/yu/zu`、`xsu/ysu/zsu`、`x/y/z`、`xs/ys/zs`。
优先选取未包裹坐标。当前不使用 `ix/iy/iz`；若只有包裹坐标和 image flags，建议重新导出 `xu/yu/zu`。
XYZ/ASE 导入只能校验元素序列，无法识别相同元素原子的重排，必须由数据提供者保证身份一致。

包裹坐标按最近镜像解包裹。它要求每一原子每一方向在相邻保存帧之间的真实位移严格小于半个盒长。
包裹轨迹自身不能验证这一假设，稀疏采样存在不可恢复的混叠；恰好半盒长时程序拒绝猜测。
未包裹轨迹不会再次做周期截断。此版本不自动去质心/框架漂移，不自动删除平衡段。

## 跨引擎比较前需要补齐

温度、化学计量、缺陷浓度、晶胞尺寸、积分步长与输出间隔、恒温器、平衡长度、采样长度、
势函数/SK 参数/泛函、许可来源均需要记录。DFTB+ 的 Li/O/Cl/Br 参数适用性和经典势准确性需要另行验证。
仅统一坐标格式不能消除引擎的系统偏差。

## 表示和评估

现有统一表示为每个 Li 的最近 H 步笛卡尔位移，预测下一步三分量位移。
它不含邻域、元素嵌入、能量或旋转等变性；不是通用原子环境表示，也不是 ML 势。
先按原始帧切分训练/验证/测试，再分别构造窗口，所有原子使用同一帧边界。
岭回归归一化只来自训练集。小 Transformer 依据验证误差选取 epoch，测试只用于最终评估。
超参数搜索应固定协议并保留最终测试集；正式工作应增加独立轨迹/温度/材料分组留出。

MSD 使用全部时间原点平均，`D = slope/6`，Å²/ps 到 cm²/s 的转换因子为 `1e-4`。
拟合区间按 lag 索引 `[start,stop)` 明确给出；负斜率会标记无效并输出空 D。
高 R² 不能证明进入扩散区间；未计算置信区间、电导率或 Haven 比。

官方接口参考（2026-11-05 核对）：

- [ASE 文件 I/O](https://ase-lib.org/ase/io/io.html)
- [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [PyTorch TransformerEncoderLayer](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html)
