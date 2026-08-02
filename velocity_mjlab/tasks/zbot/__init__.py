"""Register the Zbot velocity tasks."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import make_env_cfg
from .rl_cfg import make_rl_cfg


register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-Zbot",
    env_cfg=make_env_cfg(rough=True),
    play_env_cfg=make_env_cfg(rough=True, play=True),
    rl_cfg=make_rl_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Zbot",
    env_cfg=make_env_cfg(rough=False),
    play_env_cfg=make_env_cfg(rough=False, play=True),
    rl_cfg=make_rl_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
