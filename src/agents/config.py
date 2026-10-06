
from dataclasses import dataclass, fields
from pathlib import Path

import yaml

@dataclass
class PPOConfig:
    # env
    env_id: str = "CartPole-v1"
    
    # policy
    policy: str = "auto"
    hidden_sizes: tuple = (64, 64) #number of neurons in the hidden layers
    
    # rollout
    total_steps: int = 150_000
    rollout_steps: int = 2048 #steps collected per policy update
    epochs: int = 10 #passes over each rollout
    minibatch_size: int = 64
    lr: float = 3e-4
    
    # rewards
    gamma: float = 0.99 #disciunt factor
    gae_lambda: float = 0.95 #GAE bias
    
    # PPO objectives
    clip_eps: float = 0.2 #PPO clipping range
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    
    seed: int = 0
    log_every: int = 1
    save_path: str = "artifacts/policy.pt"
    
    def __post_init__(self):
        self.hidden_sizes = tuple(self.hidden_sizes)
        
    @classmethod
    def from_yaml(cls, path: str | Path) -> "PPOConfig":
        raw = yaml.safe_load(Path(path).read_text()) or {}
        unkown = set(raw) - {f.name for f in fields(cls)}
        if unkown:
            raise ValueError(f"unknown key in {path}: {sorted(unkown)}")
        
        return cls(**raw)