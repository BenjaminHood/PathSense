from src.envs.pathsense_env import PathSenseEnv
from src.envs.config import EnvConfig
from src.agents.ppo import train
from src.agents.config import PPOConfig

env = PathSenseEnv(EnvConfig())
policy, returns = train(
    PPOConfig(total_steps=1000000, env_id="PathSense-v0", save_path="artifacts/nav_policy.pt"),
    env=env, block=True,
)