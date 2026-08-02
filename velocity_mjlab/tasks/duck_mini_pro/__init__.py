"""Register the Duck Mini Pro velocity task."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from .env_cfg import make_env_cfg
from .rl_cfg import make_rl_cfg

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Duck-Mini-Pro",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=make_rl_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-Duck-Mini-Pro-Website",
    env_cfg=make_env_cfg(website=True),
    play_env_cfg=make_env_cfg(play=True, website=True),
    rl_cfg=make_rl_cfg(website=True),
    runner_cls=VelocityOnPolicyRunner,
)
