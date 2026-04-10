from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  duck_mini_flat_env_cfg,
  duck_mini_rough_env_cfg,
)
from .rl_cfg import duck_mini_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Duck-Mini",
  env_cfg=duck_mini_rough_env_cfg(),
  play_env_cfg=duck_mini_rough_env_cfg(play=True),
  rl_cfg=duck_mini_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Duck-Mini",
  env_cfg=duck_mini_flat_env_cfg(),
  play_env_cfg=duck_mini_flat_env_cfg(play=True),
  rl_cfg=duck_mini_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
