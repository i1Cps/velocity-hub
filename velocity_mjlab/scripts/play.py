"""Play velocity-mjlab tasks with keyboard velocity control.

Keyboard bindings:
  UP / DOWN       lin_vel_x  +/- 0.2
  J / L           lin_vel_y  +/- 0.2
  LEFT / RIGHT    ang_vel_z  +/- 0.2
  K               zero all commands
  ENTER           reset environment (also zeros commands)
"""

from mjlab.scripts.play import PlayConfig
from mjlab.viewer import NativeMujocoViewer
from mjlab.viewer.native.keys import (
    KEY_DOWN,
    KEY_J,
    KEY_K,
    KEY_L,
    KEY_LEFT,
    KEY_RIGHT,
    KEY_UP,
)

STEP = 0.2


class VelocityKeyboardViewer(NativeMujocoViewer):
    """NativeMujocoViewer with arrow-key velocity command control."""

    def __init__(self, env, policy, **kwargs):
        super().__init__(env, policy, **kwargs)
        # [lin_vel_x, lin_vel_y, ang_vel_z]
        self._kb_cmd = [0.0, 0.0, 0.0]
        self._kb_active = True
        self.user_key_callback = self._velocity_key_callback

    def _velocity_key_callback(self, key: int) -> None:
        if key == KEY_UP:
            self._kb_cmd[0] += STEP
        elif key == KEY_DOWN:
            self._kb_cmd[0] -= STEP
        elif key == KEY_J:
            self._kb_cmd[1] += STEP
        elif key == KEY_L:
            self._kb_cmd[1] -= STEP
        elif key == KEY_LEFT:
            self._kb_cmd[2] += STEP
        elif key == KEY_RIGHT:
            self._kb_cmd[2] -= STEP
        elif key == KEY_K:
            self._kb_cmd[:] = [0.0, 0.0, 0.0]
        else:
            return
        print(f"cmd: lin_x={self._kb_cmd[0]:+.2f}  lin_y={self._kb_cmd[1]:+.2f}  ang_z={self._kb_cmd[2]:+.2f}")

    def _inject_keyboard_command(self) -> None:
        """Write keyboard command into the velocity command term."""
        cmd_term = self.env.unwrapped.command_manager.get_term("twist")
        if cmd_term is None:
            return
        cmd_term.vel_command_b[self.env_idx, 0] = self._kb_cmd[0]
        cmd_term.vel_command_b[self.env_idx, 1] = self._kb_cmd[1]
        cmd_term.vel_command_b[self.env_idx, 2] = self._kb_cmd[2]

    def _execute_step(self) -> bool:
        result = super()._execute_step()
        if result:
            if self.env.unwrapped.episode_length_buf[self.env_idx] == 0:
                self._kb_cmd[:] = [0.0, 0.0, 0.0]
                print("cmd: lin_x= 0.00  lin_y= 0.00  ang_z= 0.00  [reset]")
            self._inject_keyboard_command()
        return result

    def reset_environment(self) -> None:
        self._kb_cmd[:] = [0.0, 0.0, 0.0]
        super().reset_environment()
        self._inject_keyboard_command()


def main():
    import sys

    import torch
    import tyro
    from dataclasses import asdict

    import mjlab
    import mjlab.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import (
        list_tasks,
        load_env_cfg,
        load_rl_cfg,
        load_runner_cls,
    )
    from mjlab.utils.os import get_wandb_checkpoint_path
    from mjlab.utils.torch import configure_torch_backends

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    args = tyro.cli(
        PlayConfig,
        args=remaining_args,
        default=PlayConfig(),
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(chosen_task, play=True)
    agent_cfg = load_rl_cfg(chosen_task)

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    if args.agent == "zero":
        action_shape = env.unwrapped.action_space.shape
        policy = lambda obs: torch.zeros(action_shape, device=env.unwrapped.device)
    elif args.agent == "random":
        action_shape = env.unwrapped.action_space.shape
        policy = lambda obs: 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1
    else:
        from pathlib import Path

        if args.checkpoint_file is not None:
            resume_path = Path(args.checkpoint_file)
        elif args.wandb_run_path is not None:
            log_root_path = (
                Path("logs") / "rsl_rl" / agent_cfg.experiment_name
            ).resolve()
            resume_path, _ = get_wandb_checkpoint_path(
                log_root_path, Path(args.wandb_run_path), args.wandb_checkpoint_name
            )
        else:
            raise ValueError("Provide --wandb-run-path or --checkpoint-file")

        runner_cls = load_runner_cls(chosen_task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
        )
        policy = runner.get_inference_policy(device=device)

    print("\n--- Keyboard Velocity Control ---")
    print("  UP/DOWN      lin_vel_x  +/- 0.2")
    print("  J/L          lin_vel_y  +/- 0.2")
    print("  LEFT/RIGHT   ang_vel_z  +/- 0.2")
    print("  K            zero commands")
    print("  ENTER        reset env + commands")
    print("---------------------------------\n")

    VelocityKeyboardViewer(env, policy).run()
    env.close()


if __name__ == "__main__":
    main()
