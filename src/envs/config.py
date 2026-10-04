from dataclasses import dataclass

@dataclass
class EnvConfig:
    arena_size: float = 20.0
    max_steps: int = 500
    dt: float = 0.1
    walker_speed: float = 1.2
    fov_deg: float = 120.0
    goal_radius: float = 0.8
    collision_margin: float = 0.2
    near_miss_margin: float = 1.0
    n_obstacles_range: tuple = (4, 10)
    obstacle_radius_range: tuple = (0.3, 0.9)
    reaction_delay_range: tuple = (1, 4)
    compliance_range: tuple = (0.7, 1.0)
    heading_noise_std_range: tuple = (0.01, 0.05)
    reward_step: float = -0.01
    reward_cue: float = -0.01
    reward_repeat_cue: float = -0.1
    reward_progress_scale: float = 1.0
    reward_collision: float = -10.0
    reward_near_miss: float = -0.5
    reward_goal: float = 50.0