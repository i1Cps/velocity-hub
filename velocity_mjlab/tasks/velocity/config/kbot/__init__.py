from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfgs import (
  kbot_flat_env_cfg,
  kbot_rough_env_cfg,
)
from .rl_cfg import kbot_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Velocity-Rough-Kbot",
  env_cfg=kbot_rough_env_cfg(),
  play_env_cfg=kbot_rough_env_cfg(play=True),
  rl_cfg=kbot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Velocity-Flat-Kbot",
  env_cfg=kbot_flat_env_cfg(),
  play_env_cfg=kbot_flat_env_cfg(play=True),
  rl_cfg=kbot_ppo_runner_cfg(),
  runner_cls=VelocityOnPolicyRunner,
)
