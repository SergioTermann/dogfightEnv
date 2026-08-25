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

## 目录结构

- `dogfightEnv.py` / `oneVSoneEnv.py` / `reoneVSoneEnv.py` / `twoVStwo.py` — Gym 风格仿真环境（1v1 / 2v2）
- `IA_enemy_env.py` — AI 敌机环境
- `human_expert_env.py` / `human_expert_state.py` — 人类专家数据采集相关
- `controller_env.py` — 操控接口环境
- `dogfight_sandbox_hg2/` — Harfang3D 仿真沙盒
  - `source/jsbsim_flight_model.py` — JSBSim 六自由度飞行解算封装
  - `bin/pylibs/` — jsbsim + numpy（嵌入式 Python 3.8 运行时依赖）
  - `tools/test_jsbsim_physics.py` — 物理符号校准/回归测试（14 项）
  - 启动方式：`start.bat`（手动选 Network mode）或 `bin\python\python.exe source\main.py auto_network`（跳过菜单直达网络模式，测试用）

## 运行

1. 启动沙盒：`dogfight_sandbox_hg2\start.bat`，菜单选择 "Network mode"
2. 连接环境：`oneVSoneEnv(host, port=50888)`

## 注意

`dogfight_sandbox_hg2/source/assets/` 与 `assets_compiled/`（约 1.6 GB 的模型/贴图资源）未纳入版本管理。如需完整资源请通过网盘或其他方式分发，或使用 [Git LFS](https://git-lfs.com/)。
