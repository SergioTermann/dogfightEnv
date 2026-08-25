# dogfightEnv

基于 [Harfang3D](https://harfang3d.com/) `dogfight_sandbox` 的空战强化学习仿真环境。

## 目录结构

- `dogfightEnv.py` / `oneVSoneEnv.py` / `reoneVSoneEnv.py` / `twoVStwo.py` — Gym 风格仿真环境（1v1 / 2v2）
- `IA_enemy_env.py` — AI 敌机环境
- `human_expert_env.py` / `human_expert_state.py` — 人类专家数据采集相关
- `controller_env.py` — 操控接口环境
- `dogfight_sandbox_hg2/` — Harfang3D 仿真沙盒（bin 内含可执行运行时，详见其内部 README）

## 注意

`dogfight_sandbox_hg2/source/assets/` 与 `assets_compiled/`（约 1.6 GB 的模型/贴图资源）未纳入版本管理。如需完整资源请通过网盘或其他方式分发，或使用 [Git LFS](https://git-lfs.com/)。
