# LiftSim 仿真项目

用于构建 18 层 / 2 部电梯的离散时间仿真环境，支持规则调度与后续 RL 扩展。

## 开发环境
- Python 3.8+
- Conda：`conda activate rl`

## 安装
```
conda activate rl
pip install -e .[dev]
```
可选：`pip install -e .[rl]` 安装 gymnasium、stable-baselines3 等 RL 相关依赖。

## 常用命令
- `make test`：运行 pytest 基础用例。
- `pytest -k smoke -vv`：按需运行单个/部分测试。

## 目录结构
- `core/`：仿真核心（电梯、乘客、调度器、环境、指标）。
- `gym_env/`：Gymnasium 适配器。
- `configs/`：默认与自定义场景配置。
- `tests/`：单元与回归测试。
- `experiments/`：实验脚本与记录（占位）。
- `scripts/`：工具脚本（占位）。

## 配置与复现
- 示例配置：`configs/default.yaml`，包含层数、电梯数、速度、停靠时间、spawn 概率、horizon、seed。
- 运行前确保设定 `seed`，以便复现仿真结果和测试。

## CI
GitHub Actions 流水线会执行 `pip install -e .[dev]` 与 `pytest`，保持测试通过后再提交。***
