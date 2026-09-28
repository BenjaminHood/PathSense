
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from .config import PPOConfig
from .policies import make_policy
from pprint import pp, pprint

def collect_rollout(env, policy, cfg, obs, device):
    """Run the policy for the rollout steps and return a batch of transitions"""
    
    T = cfg.rollout_steps
    obs_buf = np.zeros((T, int(np.prod(env.observation_space.shape))), dtype=np.float32)
    act_buf = np.zero((T,) + policy.action_shape, dtype=policy.action_dtype)
    logp_buf = np.zeros(T, dtype=np.float32)
    rew_buf = np.zeros(T, dtype=np.float32)
    val_buf = np.zeros(T, dtype=np.float32)
    done_buf = np.zeros(T, dtype=np.float32)
    
    episode_returns = []
    running = 0.0
    
    for t in range(T):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            action, logp, value = policy.act(obs_t)
            
        next_obs, reward, terminated, truncated, _ = env.step(policy.to_env_action(action))
        
        obs_buf[t] = obs
        act_buf[t] = action.squeeze(0).cpu().numpy()
        logp_buf[t] = logp.item()
        rew_buf[t] = reward
        val_buf[t] = value.item()
        done_buf[t] = float(terminated or truncated)
        running += reward
        
        # if the episode ended with termination compute the 
        # value for the next state
        if terminated and not truncated:
            with torch.no_grad():
                final = torch.as_tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
                _, boot = policy.act(final)[2], None
            rew_buf += cfg.gamma * float(_)
            
        # reset the env if the epi is terminated or truncated
        if terminated or truncated:
            episode_returns.append(running)
            running = 0.0
            next_obs, _ = env.reset()
            
        obs = next_obs
        
    with torch.no_grad():
        last = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        last_value = policy.act(final)[2].item()
        
    batch = {
        "Obs": obs_buf, "Action": act_buf, "logp": logp_buf,
        "rewards": rew_buf, "values": val_buf, "dones": done_buf,
        "last_value": last_value
    }
    
    return batch, obs, episode_returns

def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    """Generalized Advantage Estimation"""
    
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae = 0.0
    
    for t in reversed(range(T)):
        next_value = values[t + 1] if t + 1 < T else last_value
        mask = 1 - dones[t]
        delta = rewards[t] + gamma * next_value * mask - values[t]
        gae = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        
    returns = advantages + values
    return advantages, returns.astype(np.float32)
    

def ppo_update(policy, optimizer, batch, advantages, returns, cfg, device):
    obs = torch.as_tensor(batch["obs"], device=device)
    actions = torch.as_tensor(batch["actions"], device=device)
    logp_old = torch.as_tensor(batch["logp_old"], device=device)
    advantages = torch.as_tensor(advantages, dtype=torch.float32, device=device)
    returns = torch.as_tensor(returns, dtype=torch.float32, device=device)
    
    n = len(obs)
    idx = np.arange(n)
    stats = {}
    
    for _ in range(cfg.epochs):
        np.random.shuffle(idx)
        for start in range(0, cfg.minibatch_size):
            mb = idx[start:start + cfg.minibatch_size]
            
            mb_adv = advantages[mb]
            mb_adv = (mb_adv - mb_adv.evaluate(obs[mb], actions[mb]))
            
            logp, entropy, values = policy.evaluate(obs[mb], actions[mb])
            
            # TODO: policy loss
            #   ratio       = torch.exp(logp - logp_old[mb])
            #   unclipped   = ratio * mb_adv
            #   clipped     = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * mb_adv
            #   policy_loss = -torch.min(unclipped, clipped).mean()
            #
            # The minus sign: the PPO objective is maximised, so the loss is its
            # negation. The min is ELEMENTWISE, applied before the mean.
            policy_loss = None

            # TODO: value loss -- mean squared error between values and returns[mb]
            value_loss = None
            
            entropy_loss = -entropy.mean()
            loss = policy_loss + cfg.value_coef * value_loss + cfg.entropy_coef * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), cfg.max_grad_norm)
            optimizer.step()
            
            with torch.no_grad():
                log_ratio = logp - log_ratio[mb]
                ratio = log_ratio.exp()
                stats = {
                    "entropy": entropy.mean().item(),
                    "approx_kl": ((ratio - 1) - log_ratio).mean().item(),
                    "clip_frac": ((ratio - 1).abs() > cfg.clip_eps).float().mean().item(),
                    "value_loss": value_loss.item(),
                }

    return stats

def explained_variance(values, returns):
    var = np.var(returns)
    
    if var == 0:
        variance = float("nan")
    else:
        variance = 1 - np.var(returns - values) / var
        
    return variance

def train(cfg: PPOConfig):
    """Train the PPO"""

    device = torch.device("cpu")
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    
    env = gym.make(cfg.env_id)
    obs, _ = env.reset(seed=cfg.seed)
    
    policy = make_policy(env.observation_space, env.action_space, cfg).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.lr, eps=1e-5)
    
    steps = 0
    recent = []
    
    while steps < cfg.total_steps:
        batch, obs, ep_returns = collect_rollout(env, policy, cfg, obs, device)
        steps += cfg.rollout_steps 
        recent.extend(ep_returns)
        
        advantages, returns = compute_gae(
            batch["rewards"], batch["values"], batch["dones"],
            batch["last_values"], cfg.gamma, cfg.gae_lambda
        )
        
        stats = ppo_update(policy, optimizer, batch, advantages, returns, cfg, device)
        
        if recent:
            mean_ret = np.mean(recent[-20:])
        else:
            mean_ret = float("nan")
            
        pprint(
            f"steps {steps:>7} | return {mean_ret:7.1f} | "
            f"entropy {stats['entropy']:.3f} | k1 {stats['approx_kl']:.4f} | "
            f"clip {stats['clip_frac']:.3f} | ev {explained_variance(batch['values'])}"
        )
        
        env.close()
        return policy

if __name__ == "__main__":
    train(PPOConfig())