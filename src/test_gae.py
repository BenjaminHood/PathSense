import numpy as np
from src.agents.ppo import compute_gae

def test_gae_reduces_to_returns_to_go():
    adv, ret = compute_gae(
        rewards=np.array([1.0, 1.0, 1.0]),
        values=np.array([0.0, 0.0, 0.0]),
        dones=np.array([0.0, 0.0, 1.0]),
        last_value=0.0, gamma=1.0, gae_lambda=1.0,
    )
    np.testing.assert_allclose(adv, [3.0, 2.0, 1.0])


def test_mask_blocks_credit_across_episode_boundary():
    # episode ends at t=1, so step 2's reward must not reach steps 0-1
    adv, _ = compute_gae(
        rewards=np.array([1.0, 1.0, 1.0]),
        values=np.array([0.0, 0.0, 0.0]),
        dones=np.array([0.0, 1.0, 0.0]),
        last_value=5.0, gamma=1.0, gae_lambda=1.0,
    )
    np.testing.assert_allclose(adv, [2.0, 1.0, 6.0])


def test_discount_and_baseline():
    adv, ret = compute_gae(
        rewards=np.array([1.0, 1.0]),
        values=np.array([2.0, 3.0]),
        dones=np.array([0.0, 0.0]),
        last_value=4.0, gamma=0.5, gae_lambda=1.0,
    )
    np.testing.assert_allclose(adv, [0.5, 0.0])
    np.testing.assert_allclose(ret, [2.5, 3.0])


def test_lambda_zero_gives_td_errors():
    rewards = np.array([1.0, 2.0])
    values = np.array([0.5, 1.0])
    adv, _ = compute_gae(rewards, values, np.array([0.0, 0.0]),
                         last_value=2.0, gamma=0.9, gae_lambda=0.0)
    expected = [1.0 + 0.9 * 1.0 - 0.5, 2.0 + 0.9 * 2.0 - 1.0]
    np.testing.assert_allclose(adv, expected)
    

