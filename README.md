# dogfightEnv

> 面向强化学习与智能指挥研究的空战仿真环境：JSBSim 六自由度飞行动力学、Harfang3D 可视化、Gym 风格接口与可插拔任务指挥官。

<div align="center">

![dogfightEnv](docs/images/readme/cover.jpg)

**飞行动力学 · 学习智能体 · 任务指挥**

[English](README.en.md) · [网络协议](dogfight_sandbox_hg2/documentation_network.md) · [问题反馈](https://github.com/SergioTermann/dogfightEnv/issues)

</div>

## 项目定位

dogfightEnv 把一个可视化空战沙盒整理成可编程的实验平台。仿真核心负责世界状态、飞机、导弹与渲染；外部客户端负责学习或任务决策；双方通过统一的 JSON over TCP 协议交互。

适合以下工作：

- 训练连续动作或离散动作的空战策略
- 评估飞行控制、导弹规避、目标选择与编队协同
- 用规则引擎或 OpenAI 兼容模型做战术任务分配
- 通过键盘 / Xbox 手柄飞行并采集专家演示
- 在 3D 画面、JSONL 日志与 Tacview ACMI 中复盘实验

## 能力概览

| 模块 | 能力 | 关键事实 |
|---|---|---|
| 飞行动力学 | JSBSim 1.2.1 | F-16 气动模型、F100-PW-229 发动机、加力与飞控 SAS；FDM 固定 1/60 s 步长 |
| 仿真引擎 | Harfang3D | 海面 / 地形 / 云层、HUD、多镜头与实时轨迹预测 |
| RL 环境 | Gym 风格 API | `oneVSone`、`twoVSone`、`ia_enemy`；参考环境为 25 维观测 / 5 维动作 |
| 训练套件 | PyTorch 原生实现 | PPO、SAC、Rainbow；归一化、checkpoint、日志与评估入口统一 |
| 任务指挥 | 外置 Commander | 规则引擎或 OpenAI 兼容 LLM；`engage / patrol / retreat / hold` |
| 网络接口 | JSON over TCP | 默认 `host:50888`；一个沙盒进程一次接受一个 TCP 客户端 |

## 系统架构

![系统架构](docs/images/readme/architecture.svg)

RL 与 Commander 是两种独立的外部控制模式。键盘和手柄属于沙盒进程内的本地输入，不占用 TCP 客户端。训练与指挥应在不同会话中运行。

## 运行模型

![运行模型](docs/images/readme/execution.svg)

RL 环境在 renderless 模式下通过 `update_scene` 推进仿真步；Commander 只轮询状态和下发变化后的任务，不接管仿真时钟。这样可以分别测量策略训练和任务规划的行为。

## 快速开始

### 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11 |
| 沙盒运行时 | 仓库自带嵌入式 Python 3.8、Harfang、JSBSim 与 numpy |
| RL / Commander | 系统 Python 3.8+；训练额外需要 `torch>=2.0` |
| GPU | 可选；CUDA 可加速训练 |
| 资源文件 | `source/assets/` 与 `assets_compiled/` 未纳入 Git，需按项目分发方式准备 |

### 1. 启动沙盒

```powershell
cd dogfight_sandbox_hg2/source
..\bin\python\python.exe main.py auto_network mission=1
```

`mission=1 / 2 / 3` 对应 1v1 / 2v2 / 3v3 网络任务。窗口左上角显示实际 `HOST / PORT`，默认端口为 `50888`。

### 2. 接入 Gym 环境

```python
from oneVSoneEnv import oneVSoneEnv

env = oneVSoneEnv(host="192.168.1.103", port="50888", rendering=True)
obs = env.reset()
action = env.action_space.sample()  # [roll, pitch, yaw, thrust, fire]
obs, reward, done, info = env.step(action)
```

环境文件位于仓库根目录：`oneVSoneEnv.py`、`twoVStwo.py`、`IA_enemy_env.py`、`dogfightEnv.py`、`human_expert_env.py`。

### 3. 启动任务指挥官

在第二个终端执行：

```powershell
cd llm_commander
..\dogfight_sandbox_hg2\bin\python\python.exe commander.py
```

默认配置使用无外部依赖的规则引擎。常用参数：

```powershell
python commander.py --dry-run       # 只打印决策，不下发命令
python commander.py --once          # 只执行一轮决策
python commander.py --duration 60   # 运行 60 秒后退出
```

要启用 LLM，将 `llm_commander/config.json` 的 `engine` 设为 `llm` 并填写 `llm.api_key`。`api_base` 使用 OpenAI 兼容的 chat completions 接口，失败时自动保留上一轮方案。决策写入 `llm_commander/decisions.jsonl`。

## RL 训练套件

安装训练依赖：

```powershell
pip install -r requirements-train.txt
```

启动沙盒后，在另一个终端运行：

```powershell
python -m training.train --algo ppo --env oneVSone --timesteps 500000
python -m training.train --algo sac --env oneVSone --timesteps 1000000
python -m training.train --algo rainbow --env oneVSone --timesteps 1000000
```

加载 checkpoint 观看：

```powershell
python -m training.enjoy --model checkpoints/ppo_oneVSone/model_best.pt --episodes 3
```

训练产物写入 `checkpoints/<algo>_<env>/`：`model_best.pt`、`model_final.pt`、归一化状态和 `log.jsonl`。使用 `--set key=value` 覆盖超参数，使用 `--device cpu/cuda` 指定设备。

| 算法 | 动作空间 | 实现要点 |
|---|---|---|
| PPO | 连续 Box | GAE、clip objective、高斯策略 |
| SAC | 连续 Box | Twin Q、自动温度、软更新 |
| Rainbow | 离散网格 | n-step、Double Q、Dueling、NoisyNet、PER、C51 |

## 物理与预测

- 所有机型目前映射到 JSBSim F-16 数据；映射表见 `dogfight_sandbox_hg2/source/jsbsim_flight_model.py`。
- `dogfight_sandbox_hg2/config.json` 可在 `jsbsim` 与 `legacy` 物理引擎之间切换。
- 轨迹预测默认外推未来 10 秒，配置项为 `FlightPrediction.enabled / horizon_s / steps`。
- 导弹保留沙盒原有比例导引逻辑；`get_plane_state` 增加 `physics_engine` 字段。

## 仿真画面

![仿真画面](docs/images/readme/simulation.jpg)

## 操控与视角

键盘控制：`↑ ↓ ← →` 俯仰 / 滚转，`Home / End` 油门，`Space` 加力，`Enter` 机炮，`F1` 导弹，`T` 换目标，`G` 起落架，`B / N` 减速板，`C / V` 襟翼，`I` IA，`A` 自动驾驶，`E` 简化操纵。

小键盘视角：`2 / 8 / 4 / 6` 尾后 / 前方 / 左侧 / 右侧，`5` 卫星，`3` 座舱，`1` 切换跟拍目标，`Insert / PageUp` 调整视野。完整手柄映射见 [gamepad_mapping.svg](docs/images/gamepad_mapping.svg)。

## 测试

```powershell
cd dogfight_sandbox_hg2
bin\python\python.exe tools/test_jsbsim_physics.py
bin\python\python.exe tools/test_llm_commander.py
python tools/test_training_smoke.py
```

此外可运行 `python -m training.tests.test_algos_toy`，在无沙盒条件下验证三种算法的收敛与 checkpoint 往返。

## 工程边界

- 单个沙盒进程一次只接受一个 TCP 客户端，因此 Commander 与 RL 环境不能同时连接同一实例。
- 机型暂时共用 F-16 气动数据，其他机型数据待补充。
- 大型模型与贴图目录不进入 Git，建议使用 Git LFS 或独立资源包分发。

## 开发路线

- [ ] 通过 Git LFS 提供完整资源包
- [ ] 支持 Commander 与 RL 客户端并行连接
- [ ] 增加更多机型的 JSBSim 气动数据
- [ ] 增加标准化评测任务与可复现实验配置

## 许可证与致谢

`dogfight_sandbox_hg2/` 源自 [harfang3d/dogfight-sandbox-hg2](https://github.com/harfang3d/dogfight-sandbox-hg2)，使用 GPL-3.0。感谢 Harfang Technologies 与 [mrwangyou/DBRL](https://github.com/mrwangyou/DBRL)。

本项目使用 [Harfang3D](https://harfang3d.com/)、[JSBSim](https://github.com/JSBSim-Team/jsbsim) 与 [PyTorch](https://pytorch.org/)。
