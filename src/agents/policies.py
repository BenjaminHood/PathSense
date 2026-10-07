
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from torch.distributions import Categorical, Normal

def mlp(in_dim: int, hidden_sizes, out_dim: int) -> nn.Sequential:
    layers = []
    last = in_dim
    
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), nn.Tanh()]
        last = h
        
    layers.append(nn.Linear(last, out_dim))
    
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    
    
    is_discrete: bool
    action_shape: tuple
    action_dtype: type
    
    def act(self, obs):
        raise NotImplementedError
    
    def evaluate(self, obs, actions):
        raise NotImplementedError
    
    def to_env_action(self, action: torch.Tensor):
        
        raise NotImplementedError
    
class CategoricalPolicy(ActorCritic):
    """Discrete Actions"""
    
    is_discrete = True
    
    def __init__(self, obs_dim: int, n_actions: int, hidden_sizes=(64, 64)):
        super().__init__()
        self.policy_net = mlp(obs_dim, hidden_sizes, n_actions)
        self.value_net = mlp(obs_dim, hidden_sizes, 1)
        self.action_shape = ()
        self.action_dtype = np.int64
        
    def _dist(self, obs):
        return Categorical(logits=self.policy_net(obs)) #retuen categorical probs
    
    def act(self, obs):
        dist = self._dist(obs)
        action = dist.sample()
        
        return action, dist.log_prob(action), self.value_net(obs).squeeze(-1)
    
    def evaluate(self, obs, actions):
        dist = self._dist(obs)
        
        return dist.log_prob(actions), dist.entropy(), self.value_net(obs).squeeze(-1)
    
    def to_env_action(self, action):
        return int(action.item())
    
class GaussianPolicy(ActorCritic):
    """Continous actions"""
    
    is_discrete = False  
    
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes=(64, 64), init_log_std = 0.5):
       super().__init__()
       self.policy_net = mlp(obs_dim, hidden_sizes, act_dim)
       self.value_net = mlp(obs_dim, hidden_sizes, 1)
       self.log_std = nn.Parameter(torch.full((act_dim), float(init_log_std)))
       self.action_shape = (act_dim)
       self.action_dtype = np.int32
       
    def _dist(self, obs):
        return Normal(self.policy_net(obs), self.log_std.exp())
    
    def act(self, obs):
        dist = self._dist(obs)
        action = dist.sample()
        
        # sum over action dims
        return action, dist.log_prob(action).sum(-1), self.value_net(obs).squeeze(-1)
    
    def evaluate(self, obs, actions):
        dist = self._dist(obs)
        
        return (
            dist.log_prob(actions).sum(-1),
            dist.entropy().sum(-1),
            self.value_net(obs).squeeze(-1)
        )
    
    def to_env_action(self, action):
        return action.squeeze(0).cpu().numpy()
    
def make_policy(obs_space, action_space, cfg) -> ActorCritic:
    obs_dim = int(np.prod(obs_space.shape))
    
    kind  = cfg.policy
    if kind == "auto":
        if isinstance(action_space, gym.spaces.Discrete):
            kind = "categorical"
        else:
            kind = "gaussian"
            
    if kind == "categorical":
        return CategoricalPolicy(obs_dim, action_space.n, cfg.hidden_sizes)
    if kind == "gaussian":
        return GaussianPolicy(obs_dim, int(np.prod(action_space.shape)), cfg.hidden_sizes)
    raise ValueError(f"unkown policy type: {cfg.policy}")