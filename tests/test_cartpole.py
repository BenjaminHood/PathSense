import pytest
import numpy as np

from src.agents.ppo import train
from src.agents.config import PPOConfig

@pytest.mark.slow
def test_ppo_solves_cartpole():
    policy, returns = train(PPOConfig(total_steps=60_000, seed=0))
    assert np.mean(returns[-20:]) > 400