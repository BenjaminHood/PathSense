from src.envs.pathsense_env import PathSenseEnv
from src.envs.config import EnvConfig
from src.agents.ppo import train
from src.agents.config import PPOConfig

env = PathSenseEnv(EnvConfig())
policy, returns = train(PPOConfig(total_steps=200_000), env=env)