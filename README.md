# dogfightEnv

基于 [Harfang3D](https://harfang3d.com/) `dogfight_sandbox` 的空战强化学习仿真环境。

## 飞行动力学：JSBSim 六自由度解算

飞机的力学解算已从沙盒原版的简化模型替换为 **JSBSim 1.2.1**（F-16A 气动数据，含 F100-PW-229 发动机、大气模型、起落架）。要点：

- 所有机型统一使用 F-16 气动模型（其余机型无开源 JSBSim 数据，映射表见 `dogfight_sandbox_hg2/source/jsbsim_flight_model.py`）
- 真实飞控（SAS）：中立杆=保持姿态；偏航杆量联动滚转（协调转弯）；迎角/航迹包线保护防失速；杆量回中自动改平
- 导弹仍用原版比例导引物理
- 网络协议、动作/观测空间与旧版完全兼容，`get_plane_state` 新增 `physics_engine` 字段（jsbsim/legacy）
- 引擎切换：`dogfight_sandbox_hg2/config.json` → `"Physics": {"engine": "jsbsim"}`（改为 `"legacy"` 回退原版物理）
- FDM 固定 1/60 步长，renderless 客户端驱动模式下每次 `update_scene` 精确推进一步（确定性）

## 画面内未来轨迹预测

沙盒 3D 视图直接绘制每架飞机未来 10 秒的预测航线（友军绿色 / 敌军红色折线，每 5 秒一个十字标记 + `+Ns` 标签），基于当前转弯率/俯仰率/加速度做指数衰减的运动学外推。配置：`config.json` → `"FlightPrediction": {"enabled": true, "horizon_s": 10, "steps": 20}`。

## 大模型任务分配指挥官（LLM Mission Commander）

`llm_commander/` 提供一个外置"空中指挥官"进程：通过 TCP/IP（:50888）观察全局战况，周期性为己方（默认红方）每架飞机分配任务并经网络命令执行，任务标签实时悬浮显示在 3D 画面中（如 `ENNEMY_1 ENGAGE ALLY_1`）。

任务词表（映射到沙盒已有命令原语）：

- `engage(target)`：指定目标攻击（`SET_TARGET_ID` → `ACTIVATE_IA`，先定目标再激活避免随机选目标）
- `patrol(heading/altitude/speed)`：自动驾驶巡航待战
- `retreat`：朝远离最近敌机的航向高速脱离
- `hold`：保持现状

决策引擎可插拔（`llm_commander/config.json` 的 `engine` 字段）：

- `"rule"`（默认）：内置规则引擎（最近敌 1v1 配对去冲突、残血/弹尽脱离、无敌可打巡逻），无需任何 API key
- `"llm"`：OpenAI 兼容 chat/completions 端点（默认对接智谱 `open.bigmodel.cn`，`llm.api_key` 填 key 即启用；DeepSeek/OpenAI/本地 vLLM 等改 `api_base` 即可），中文指挥官提示词 + 严格 JSON 输出校验（幻觉机名/非法目标自动丢弃，失败沿用上轮分配）

用法：

```bash
# 1. 启动 2v2 网络任务沙盒（mission=1/2/3 对应 1v1/2v2/3v3）
cd dogfight_sandbox_hg2/source && ../bin/python/python.exe main.py auto_network mission=2

# 2. 启动指挥官（另开一个终端；嵌入式或系统 Python 均可）
cd llm_commander && ../dogfight_sandbox_hg2/bin/python/python.exe commander.py
#    可选参数：--dry-run（只打印不下发）、--once（单轮决策）、--duration N（N 秒后退出）
```

配置要点（`llm_commander/config.json`）：`side`（allies/ennemies）、`decision_period_s`（决策周期，默认 10 s，战损事件触发立即重判）、`blue_ia`（演示时给蓝方开内置 IA 陪练）。决策记录写入 `llm_commander/decisions.jsonl`。指挥官不接管仿真步进，沙盒自由运行，LLM 延迟不影响帧率。

端到端测试（自动起沙盒跑完整链路）：`dogfight_sandbox_hg2\bin\python\python.exe tools\test_llm_commander.py`

## 目录结构

- `dogfightEnv.py` / `oneVSoneEnv.py` / `reoneVSoneEnv.py` / `twoVStwo.py` — Gym 风格仿真环境（1v1 / 2v2）
- `IA_enemy_env.py` — AI 敌机环境
- `human_expert_env.py` / `human_expert_state.py` — 人类专家数据采集相关
- `controller_env.py` — 操控接口环境
- `llm_commander/` — 大模型任务分配指挥官（commander.py 主循环 / tactician.py 决策引擎 / config.json）
- `dogfight_sandbox_hg2/` — Harfang3D 仿真沙盒
  - `source/jsbsim_flight_model.py` — JSBSim 六自由度飞行解算封装（含 predict_path 轨迹预测）
  - `bin/pylibs/` — jsbsim + numpy（嵌入式 Python 3.8 运行时依赖）
  - `tools/test_jsbsim_physics.py` — 物理符号校准/回归测试（14 项）
  - `tools/test_llm_commander.py` — 指挥官端到端测试（12 项）
  - 启动方式：`start.bat`（手动选 Network mode）或 `bin\python\python.exe source\main.py auto_network [mission=N]`（跳过菜单直达网络模式，mission=1/2/3 对应 1v1/2v2/3v3）

## 运行

1. 启动沙盒：`dogfight_sandbox_hg2\start.bat`，菜单选择 "Network mode"
2. 连接环境：`oneVSoneEnv(host, port=50888)`

## 注意

`dogfight_sandbox_hg2/source/assets/` 与 `assets_compiled/`（约 1.6 GB 的模型/贴图资源）未纳入版本管理。如需完整资源请通过网盘或其他方式分发，或使用 [Git LFS](https://git-lfs.com/)。
