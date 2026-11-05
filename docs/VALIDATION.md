# 第一版验证记录

日期：2026-11-05；设备架构：arm64；Python：系统 Python 3.9.6。

已实际执行：

```bash
python3 -m unittest discover -s tests -v
python3 scripts/run_demo.py
PYTHONPYCACHEPREFIX="$PWD/work/pycache" python3 -m compileall -q src scripts tests
```

14 项测试全部通过；标准库演示完成，产生完整实验记录、模型参数、MSD CSV 和测试预测 CSV。
显式设置编译缓存目录是为了适配受限执行环境；普通终端可直接运行 compileall。

默认 seed=42 的**合成数据**测试集下一步位移 RMSE（每坐标分量）：

| 基线 | RMSE / Å |
| --- | ---: |
| 零位移 | 0.045709 |
| 上一帧位移延续 | 0.038217 |
| 岭回归 | 0.035000 |

测试覆盖：两方向周期跨越、半盒长歧义拒绝、未包裹长跳跃保持、解析恒速 MSD、扩散单位转换、
无效输入、原始帧无交叉切分、已知线性映射、误差指标、过短数据拒绝、LAMMPS ID 排序、
非均匀时间/原子身份变化拒绝、XYZ 晶胞要求、端到端可重复性和结果覆盖保护。

尚未执行：ASE/VASP 实际数据导入、PyTorch Transformer 训练、MPS 训练、GitHub 远端 CI。
原生 LAMMPS/XYZ 读取用受控测试夹具验证，尚未用真实电解质轨迹交叉核对。
本次环境限制创建 `.git` 元数据，需在自己的终端执行 README 中的 Git 初始化命令。
这些结果仅验证工程流程，不代表真实材料输运性质或跨体系模型精度。
