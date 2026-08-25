try:
    from src.environments.dogfightEnv.dogfightEnv import DogfightEnv
    from src.environments.dogfightEnv.oneVSoneEnv import oneVSoneEnv
except:
    from gym.envs.dogfightEnv.dogfightEnv import DogfightEnv
    from gym.envs.dogfightEnv.oneVSoneEnv import oneVSoneEnv
    from gym.envs.dogfightEnv.twoVStwo import twoVStwo
    from gym.envs.dogfightEnv.human_expert_env import human_expert_env
    from gym.envs.dogfightEnv.human_expert_state import human_expert_state_env
    from gym.envs.dogfightEnv.IA_enemy_env import IA_enemy_env
    from gym.envs.dogfightEnv.reoneVSoneEnv import reoneVSoneEnv