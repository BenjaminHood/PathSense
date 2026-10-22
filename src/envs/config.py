from dataclasses import dataclass

@dataclass
class EnvConfig:
    # No goal: the walker has their own intended path, and the agent's job is to
    # get them past obstacles and back onto it with as few cues as possible.
    arena_size: float = 20.0
    max_steps: int = 300                    # fixed-length episodes (~30 s)
    dt: float = 0.1
    dt_range: tuple | None = (0.08, 0.15)   # per-episode control step; live depth runs ~7-8 Hz
    stale_ranges_prob: float = 0.3          # chance a step reuses last step's ranges (slow depth)
    start_stopped_prob: float = 0.3         # episodes that begin stopped, so it learns to restart
    walker_speed: float = 1.2
    fov_deg: float = 60.0
    collision_margin: float = 0.2
    near_miss_margin: float = 0.6           # only penalised while closing in on an obstacle
    n_obstacles_range: tuple = (4, 10)
    obstacle_radius_range: tuple = (0.3, 0.9)
    reaction_delay_range: tuple = (1, 4)
    compliance_range: tuple = (0.7, 1.0)
    heading_noise_std_range: tuple = (0.01, 0.05)
    turn_deg: float = 25.0                  # "bear left/right"; kept under half the FOV so the
                                            #   walker turns into space the camera has already seen

    # the walker's own intentions (their own turns, not caused by cues)
    intent_change_steps_range: tuple = (60, 200)   # steps between their own turns
    intent_turn_deg_range: tuple = (30.0, 90.0)    # size of those turns, either side
    wall_lookahead_m: float = 3.0                  # they turn away before walking into a wall

    # reward shaping
    reward_step: float = 0.0
    reward_progress_scale: float = 1.0      # per metre walked along their intended path
    reward_cue: float = -0.03               # every cue costs the user attention
    reward_repeat_cue: float = -0.1
    reward_collision: float = -10.0
    reward_near_miss: float = -0.5
    reward_stopped: float = -0.03           # per step standing still, so stopping isn't free
