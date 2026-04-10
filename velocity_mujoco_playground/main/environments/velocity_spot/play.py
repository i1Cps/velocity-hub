import json
import mujoco
from mujoco_playground import registry as mp_registry
from mujoco_playground._src import mjx_env
from jax import numpy as jp
from mujoco import mjx
import jax
from mujoco.mjx._src import math
from brax.io import model as network_model
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from typing import Any, Tuple
import glfw
from main.environments.velocity_spot.env import VelocitySpot

# Environment to help the inference engine construct observation and evaluate the state for a control policy
class VelocitySpot_Play:
    def __init__(self):
        options_path = "main/environments/velocity_spot/options.json"
        xml_path     = "main/environments/velocity_spot/model.xml"

        # Grab the settings in options.json
        with open(options_path) as f:
            options = json.load(f)

        self._mj_model = mujoco.MjModel.from_xml_path(xml_path)
        self.ctrl_dt   = options["mjx"]["ctrl_dt"]
        self.sim_dt    = options["mjx"]["sim_dt"]

        self._mj_model.opt.timestep = self.sim_dt
        self.skipped_frames         = self.ctrl_dt / self.sim_dt

        training_env = _make_training_env(options["mjx"])
        network_factory = options["ppo"]["network_factory"]

        # Create agent brain
        brain_neurons = network_model.load_params("models/velocity_spot/model")
        brain_input = ppo_networks.make_ppo_networks(
            observation_size           = training_env.observation_size,
            action_size                = training_env.action_size,
            policy_hidden_layer_sizes  = tuple(network_factory["policy_hidden_layer_sizes"]),
            value_hidden_layer_sizes   = tuple(network_factory["value_hidden_layer_sizes"]),
            preprocess_observations_fn = running_statistics.normalize,
        )
        make_brain_function = ppo_networks.make_inference_fn(brain_input)
        self._agent_brain   = make_brain_function(brain_neurons)

        # Environment constants
        self._joint_reset_noise    = 0.0
        self._agent_healthy_angle  = training_env._agent_healthy_angle
        self._agent_site_id        = training_env._agent_site_id
        self._action_scale         = training_env._action_scale
        self._start_pose           = training_env._start_pose
        self._start_joints         = self._start_pose[7:]

        self._command_range_linvel_x   = training_env._command_range_linvel_x
        self._command_range_linvel_y   = training_env._command_range_linvel_y
        self._command_range_angvel_yaw = training_env._command_range_angvel_yaw
        self._command_step_linvel_x    = 0.10
        self._command_step_linvel_y    = 0.10
        self._command_step_angvel_yaw  = 0.1
        self._command                  = jp.array([0.0,0.0,0.0])

        self._push_magnitude_range = training_env._push_magnitude_range
        self._push_interval_range  = training_env._push_interval_range
        self._enable_push          = training_env._push_enabled

        self._agent_qpos_ids = training_env._agent_qpos_ids
        self._agent_qvel_ids = training_env._agent_qvel_ids

        agent_qfrc_actuator_ids = []

        for dof_id in range(self._mj_model.nv):
            if dof_id in self._agent_qvel_ids:
                agent_qfrc_actuator_ids.append(dof_id)
  
        self._agent_qfrc_actuator_ids = jp.array(agent_qfrc_actuator_ids)
        self._push_interval_steps = jp.array(1, dtype=jp.int32)
        self._last_act = jp.zeros(self._mj_model.nu)
        self._unhealthy_ticks = jp.array(0)

        print("\n── Velocity Spot ──")
        print("Locomotion policy driven by velocity commands.")
        print("Random pushes are applied periodically.\n")
        print("Controls:")
        print("  Up / Down      Forward / backward velocity")
        print("  Left / Right   Yaw left / right")
        print("  J / L          Strafe left / right")
        print("  X              Stop (zero all commands)")
        print("  Space          Pause / unpause")
        print("  R              Reset\n")

    # Resets the environment to an initial state.
    def reset(self, data: mujoco.MjData, rng: jax.Array) -> Tuple[mujoco.MjData, jax.Array]:

        qpos = self._start_pose.copy()
        qvel = jp.zeros(self._mj_model.nv)

        rng, rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 6)

        # Offset the start position of the robot.
        xy_offset = jax.random.uniform(rng1, (2,), minval=-0.05, maxval=0.05)

        # Random yaw orientation of robot
        yaw = jax.random.uniform(rng2, (), minval=-jp.pi, maxval=jp.pi)
        new_orientation = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)

        # Initialise joint angles with noise
        joint_noise = self._joint_reset_noise
        new_joint_angles = qpos[7:] + jax.random.uniform(
            rng3, qpos[7:].shape, minval=-joint_noise, maxval=joint_noise
        )

        qpos = qpos.at[0:2].set(qpos[0:2] + xy_offset)
        qpos = qpos.at[3:7].set(new_orientation)
        qpos = qpos.at[7:].set(new_joint_angles)

        # Initialise global velocity
        # qvel = qvel.at[0:6].set(jax.random.uniform(rng4, (6,), minval=-0.5, maxval=0.5))

        data.qpos[self._agent_qpos_ids] = qpos
        data.qvel[self._agent_qvel_ids] = qvel
        data.qacc_warmstart[self._agent_qvel_ids] = 0.0
        data.qfrc_actuator[self._agent_qfrc_actuator_ids] = 0.0

        push_interval = jax.random.uniform(
            rng5,
            minval=self._push_interval_range[0],
            maxval=self._push_interval_range[1],
        )

        cmd = jp.array([0.0, 0.0, 0.0])
        print(f"Command: fwd={cmd[0]:.2f}  lat={cmd[1]:.2f}  yaw={cmd[2]:.2f}")
        
        self._push_interval_steps = jp.round(push_interval / self.ctrl_dt).astype(jp.int32)
        self._push_step = 0
        self._command   = cmd
        self._unhealthy_ticks = jp.array(0)

        return data, rng

    def step(self, model, data:mujoco.MjData, rng):
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        push_theta = jax.random.uniform(rng1, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            rng2,
            minval=self._push_magnitude_range[0],
            maxval=self._push_magnitude_range[1]
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= jp.mod(self._push_step + 1, self._push_interval_steps) == 0
        push *= self._enable_push
        qvel = data.qvel
        qvel[:2] += push * push_magnitude
        self._push_step += 1

        mjx_data = mjx.put_data(model, data, impl="jax")
        obs      = self.calculate_obs(mjx_data)

        # Get actions
        actions, _     = self._agent_brain(obs, rng3)
        data.ctrl      = self.preprocess_actions(actions)
        self._last_act = actions

        # Step the environment
        for frame in range(int(self.skipped_frames)):
            mujoco.mj_step(model, data)

        mjx_data = mjx.put_data(model, data, impl="warp")
        done     = self.calculate_termination(mjx_data)
        if done:
            data, rng = self.reset(data, rng4)

        return model, data, rng

    def preprocess_actions(self, actions) -> jp.ndarray:
        return (actions * self._action_scale) + self._start_joints

    def calculate_termination(self, data: mjx.Data):
        torso_orientation = data.site_xmat[self._agent_site_id].ravel()[-1]
        healthy = torso_orientation > self._agent_healthy_angle
        self._unhealthy_ticks = jp.where(healthy, 0, self._unhealthy_ticks + 1)
        return self._unhealthy_ticks > 80

    def calculate_obs(self, data: mjx.Data) -> jp.ndarray:
        linvel      = mjx_env.get_sensor_data(self._mj_model, data, "local_linvel")
        gyro        = mjx_env.get_sensor_data(self._mj_model, data, "gyro")
        gravity     = mjx_env.get_sensor_data(self._mj_model, data, "gravity")
        command     = self._command
        joint_pos   = data.qpos[self._agent_qpos_ids[7:]] - self._start_joints
        joint_vel   = data.qvel[self._agent_qvel_ids][6:]
        last_action = self._last_act

        state = jp.concatenate(
            [
                linvel,         # 3
                gyro,           # 3
                gravity,        # 3
                command,        # 3
                joint_pos,      # 12
                joint_vel,      # 12
                last_action,    # 12
            ]
        )
        return {"state": state}

    # User input for velocity commands
    def key_callback(self, keycode):
        cmd = self._command

        if keycode == glfw.KEY_UP:
            cmd = cmd.at[0].set(
                jp.clip(
                    cmd[0] + self._command_step_linvel_x,
                    self._command_range_linvel_x[0],
                    self._command_range_linvel_x[1],
                )
            )
        elif keycode == glfw.KEY_DOWN:
            cmd = cmd.at[0].set(
                jp.clip(
                    cmd[0] - self._command_step_linvel_x,
                    self._command_range_linvel_x[0],
                    self._command_range_linvel_x[1],
                )
            )
        elif keycode == glfw.KEY_J:
            cmd = cmd.at[1].set(
                jp.clip(
                    cmd[1] + self._command_step_linvel_y,
                    self._command_range_linvel_y[0],
                    self._command_range_linvel_y[1],
                )
            )
        elif keycode == glfw.KEY_L:
            cmd = cmd.at[1].set(
                jp.clip(
                    cmd[1] - self._command_step_linvel_y,
                    self._command_range_linvel_y[0],
                    self._command_range_linvel_y[1],
                )
            )
        elif keycode == glfw.KEY_LEFT:
            cmd = cmd.at[2].set(
                jp.clip(
                    cmd[2] + self._command_step_angvel_yaw,
                    self._command_range_angvel_yaw[0],
                    self._command_range_angvel_yaw[1],
                )
            )
        elif keycode == glfw.KEY_RIGHT:
            cmd = cmd.at[2].set(
                jp.clip(
                    cmd[2] - self._command_step_angvel_yaw,
                    self._command_range_angvel_yaw[0],
                    self._command_range_angvel_yaw[1],
                )
            )
        elif keycode == glfw.KEY_X:
            cmd = jp.zeros(3)
        else:
            return

        self._command = cmd
        print(f"Command: fwd={cmd[0]:.2f}  lat={cmd[1]:.2f}  yaw={cmd[2]:.2f}")


# The purist' compromise
def _make_training_env(mjx_options: dict[str, Any]) -> VelocitySpot:
    env_cfg = mp_registry.get_default_config("velocity_spot")
    env_cfg["impl"] = mjx_options["impl"]
    env_cfg["ctrl_dt"] = mjx_options["ctrl_dt"]
    env_cfg["sim_dt"] = mjx_options["sim_dt"]
    return VelocitySpot(config=env_cfg)
