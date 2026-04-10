from typing import Any, Dict, Optional, Union
from mujoco_playground._src import mjx_env
from mujoco_playground import locomotion
from mujoco.mjx._src import math

from jax import numpy as jp
import jax
import mujoco
from mujoco import mjx
from ml_collections import config_dict

def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        naconmax=80000,
        njmax=100,
        seed=14
    )

class VelocityZbot(mjx_env.MjxEnv):
    """ Environment to train the Zbot to follow velocity commands"""
    def __init__(
        self,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):

        super().__init__(config, config_overrides)
        self._xml_path  = "main/environments/velocity_zbot/model.xml"
        self._mj_model  = mujoco.MjModel.from_xml_path(self._xml_path)
        self._mj_model.opt.timestep = self.sim_dt
        self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)

        # Action scale
        self._action_scale = 0.6

        # Noise applied to robot's starting joints positions
        self._joint_reset_noise = 0.05

        # Observation noise scale
        self._joint_pos_noise_scale = 0.03
        self._joint_vel_noise_scale = 1.50
        self._gravity_noise_scale   = 0.5
        self._linvel_noise_scale    = 0.1
        self._gyro_noise_scale      = 0.2

        # Velocity command ranges
        self._command_range_linvel_x   = [-0.2, 0.5]
        self._command_range_linvel_y   = [-0.2, 0.2]
        self._command_range_angvel_yaw = [-1.0, 1.0]

        # Push settings
        self._push_enabled         = True
        self._push_interval_range  = [5.0, 10.0]
        self._push_magnitude_range = [0.0, 0.5]

        # Termination settings
        self._agent_healthy_angle = 0.34

        # Reward weights
        self._tracking_linvel_reward_weight = 2.0
        self._tracking_angvel_reward_weight = 2.00
        self._upright_reward_weight         = 1.0
        self._angvel_xy_cost_weight         = -0.15
        self._termination_cost_weight       = -100.0
        self._action_rate_cost_weight       = -0.15
        self._feet_slip_cost_weight         =  0.0
        self._foot_swing_height_cost_weight = -0.5
        self._pose_reward_weight            = 2.0
        self._target_height                 = 0.06

        # Calculate the joint ids of the robot
        agent_qpos_ids = []
        agent_qvel_ids = []
        curr_qpos_id = 0
        curr_qvel_id = 0

        for joint_id in range(self._mj_model.njnt):
            joint_name = self._mj_model.joint(joint_id).name
            if "floating_base" in joint_name:
                agent_qpos_ids.extend(range(curr_qpos_id, curr_qpos_id + 7))
                agent_qvel_ids.extend(range(curr_qvel_id, curr_qvel_id + 6))

                # Increment the id counters (7DOF, 3 positional, 4 rotational)
                curr_qpos_id += 7
                curr_qvel_id += 6
            else:
                agent_qpos_ids.append(curr_qpos_id)
                agent_qvel_ids.append(curr_qvel_id)
                curr_qpos_id += 1
                curr_qvel_id += 1

        self._agent_qpos_ids = jp.array(agent_qpos_ids)
        self._agent_qvel_ids = jp.array(agent_qvel_ids)
        self._agent_site_id = self._mj_model.site("IMU").id

        self._start_pose = jp.array(self._mj_model.keyframe("home").qpos)
        self._start_joints = jp.array(self._mj_model.keyframe("home").qpos[7:])

        # Get the feet site id's
        self._feet_site_id = jp.array([
            self._mj_model.site("left_foot").id,
            self._mj_model.site("right_foot").id
        ])

    # Resets the environment to an initial state.
    def reset(self, rng: jax.Array) -> mjx_env.State:

        qpos = self._start_pose
        qvel = jp.zeros(self.mjx_model.nv)
        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        # Offset the start position of the robot
        xy_offset = jax.random.uniform(rng1, (2,), minval=-0.05, maxval=0.05)

        # Random yaw orientation of robot
        yaw = jax.random.uniform(rng2, (), minval=-jp.pi, maxval=jp.pi)
        new_orientation = math.axis_angle_to_quat(jp.array([0.0, 0.0, 1.0]), yaw)

        # Initialise joint angles with noise
        joint_noise = self._joint_reset_noise
        new_joint_angles = qpos[7:] + jax.random.uniform(rng3, qpos[7:].shape, minval=-joint_noise, maxval=joint_noise)

        qpos = qpos.at[0:2].set(qpos[0:2] + xy_offset)
        qpos = qpos.at[3:7].set(new_orientation)
        qpos = qpos.at[7:].set(new_joint_angles)

        # Initialise global velocity
        qvel = qvel.at[0:6].set(jax.random.uniform(rng4, (6,), minval=-0.5, maxval=0.5))

        data = mjx_env.make_data(
            self.mj_model,
            qpos    = qpos,
            qvel    = qvel,
            ctrl    = jp.zeros(qpos[7:].shape),
            impl    = self.mjx_model.impl.value,
            naconmax = self._config.naconmax,
            njmax   = self._config.njmax,
        )
        data = mjx.forward(self.mjx_model, data)

        # Sample a command
        rng, command_key = jax.random.split(rng)
        command = self._sample_command(command_key)

        # Sample push interval.
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._push_interval_range[0],
            maxval=self._push_interval_range[1]
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        info = {
            "rng"                : rng,
            "step"               : 0,
            "command"            : command,
            "last_act"           : jp.zeros(self.mjx_model.nu),
            "feet_air_time"      : jp.zeros(2),
            "swing_peak"         : jp.zeros(2),
            "peak_height_sum"    : jp.zeros(()),
            "peak_height_count"  : jp.zeros(()),
            "peak_height_mean"   : jp.zeros(()),
            "first_contact"      : jp.zeros(2, dtype=bool),
            "landing_peak"       : jp.zeros(2),
            "last_contact"       : jp.zeros(2, dtype=bool),
            "push_step"          : 0,
            "push_interval_steps": push_interval_steps,
        }

        metrics = {}
        metrics["reward/tracking_linvel"] = jp.zeros(())
        metrics["reward/tracking_angvel"] = jp.zeros(())
        metrics["reward/upright"]         = jp.zeros(())
        metrics["reward/feet_air_time"]   = jp.zeros(())
        metrics["cost/action_rate"]       = jp.zeros(())
        metrics["cost/termination"]       = jp.zeros(())
        metrics["cost/ang_vel_xy"]        = jp.zeros(())
        metrics["cost/feet_slip"]         = jp.zeros(())
        metrics["cost/foot_swing_height"] = jp.zeros(())
        metrics["reward/pose"]            = jp.zeros(())
        metrics["metric/peak_height_mean"] = jp.zeros(())

        obs = self._calculate_obs(data, info)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng = state.info["rng"]
        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)

        # Handle the push step
        push_direction = jax.random.uniform(rng1, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            rng2,
            minval=self._push_magnitude_range[0],
            maxval=self._push_magnitude_range[1]
        )
        push = jp.array([jp.cos(push_direction), jp.sin(push_direction)])
        push *= (jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0)
        push *= self._push_enabled
        qvel = state.data.qvel
        qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        # Preprocess action
        preprocessed_action = action * self._action_scale
        preprocessed_action = self._start_joints + preprocessed_action

        # Step the environment
        data = mjx_env.step(self.mjx_model, state.data, preprocessed_action, self.n_substeps)

        # Handle feet information
        contacts = self._get_feet_contacts(data)
        contact_filter = contacts | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filter
        landing_peak, peak_height_mean_delta = self._update_feet_height_state(
            data, state.info, contacts, first_contact
        )
        state.info["feet_air_time"] += self.dt
        state.info["first_contact"] = first_contact
        state.info["landing_peak"] = landing_peak

        # Basic RL stuff
        obs    = self._calculate_obs(data, state.info)
        done   = self._calculate_done(data)
        reward = self._get_reward(
            data,
            action,
            state.info,
            state.metrics,
            done,
        )
        state.metrics["metric/peak_height_mean"] = peak_height_mean_delta
        reward = reward * self.dt

        # Update the information dictionary
        state.info["step"]      += 1
        state.info["push_step"] += 1
        state.info["last_act"]   = action 

        state.info["command"] = jp.where( 
            state.info["step"] > 500,
            self._sample_command(rng3),
            state.info["command"],
        )

        state.info["step"] = jp.where(
            jp.logical_or(done, state.info["step"] > 500),
            0,
            state.info["step"],
        )
        state.info["feet_air_time"] *= ~contacts
        state.info["last_contact"] = contacts
        state.info["rng"] = rng


        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _calculate_obs(self, data: mjx.Data, info: dict[str, Any]) -> mjx_env.Observation:

        rng = info["rng"]
        rng, rng1, rng2, rng3, rng4, rng5 = jax.random.split(rng, 6)

        linvel = mjx_env.get_sensor_data(self._mj_model, data, "local_linvel")
        noisy_linvel = (
            linvel
            + (2 * jax.random.uniform(rng5, shape=linvel.shape) - 1)
            * self._linvel_noise_scale
        )

        gyro = mjx_env.get_sensor_data(self._mj_model, data, "gyro")
        noisy_gyro = (
            gyro
            + (2 * jax.random.uniform(rng1, shape=gyro.shape) - 1)
            * self._gyro_noise_scale
        )

        gravity = mjx_env.get_sensor_data(self._mj_model, data, "gravity")
        noisy_gravity = (
            gravity
            + (2 * jax.random.uniform(rng2, shape=gravity.shape) - 1)
            * self._gravity_noise_scale
        )

        joint_angles = data.qpos[self._agent_qpos_ids[7:]] - self._start_joints
        noisy_joint_angles = (
            joint_angles
            + (2 * jax.random.uniform(rng3, shape=joint_angles.shape) - 1)
            * self._joint_pos_noise_scale
        )

        joint_vel = data.qvel[self._agent_qvel_ids[6:]]
        noisy_joint_vel = (
            joint_vel
            + (2 * jax.random.uniform(rng4, shape=joint_vel.shape) - 1)
            * self._joint_vel_noise_scale
        )

        state = jp.hstack([
            noisy_linvel,       # 3
            noisy_gyro,         # 3
            noisy_gravity,      # 3
            info["command"],    # 3
            noisy_joint_angles, # 18
            noisy_joint_vel,    # 18
            info["last_act"],   # 18
        ])

        accelerometer = mjx_env.get_sensor_data(self._mj_model, data, "accelerometer")
        global_angvel = mjx_env.get_sensor_data(self._mj_model, data, "global_angvel")
        root_height = data.site_xpos[self._agent_site_id][2]

        contacts = self._get_feet_contacts(data)

        privileged_state = jp.hstack([
            state,                 # 66
            gyro,                  # 3
            accelerometer,         # 3
            gravity,               # 3
            linvel,                # 3
            global_angvel,         # 3
            joint_angles,          # 18
            joint_vel,             # 18
            root_height,           # 1
            data.actuator_force,   # 18
            contacts,              # 2
            info["feet_air_time"], # 2
        ])

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _calculate_done(self, data: mjx.Data) -> jax.Array:
        # Get orientation of the robot's root body in world coordinates.
        torso_orientation   = data.site_xmat[self._agent_site_id].ravel()[-1]
        healthy_orientation = torso_orientation > self._agent_healthy_angle
        return jp.where(healthy_orientation, 0.0, 1.0)

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
    ) -> jax.Array:

        # Calculate the reward values from each reward function
        tracking_linvel = self._reward_tracking_linvel(data, info)
        tracking_angvel = self._reward_tracking_angvel(data, info)
        upright         = self._reward_upright(data)
        angvel_xy       = self._cost_angvel_xy(data)
        pose            = self._reward_pose(data, info)
        termination     = self._cost_termination(done)
        feet_slip       = self._cost_feet_slip(data)
        foot_swing_height = self._cost_foot_swing_height(info)

        # Apply reward weights
        tracking_linvel_reward = tracking_linvel * self._tracking_linvel_reward_weight
        tracking_angvel_reward = tracking_angvel * self._tracking_angvel_reward_weight
        upright_reward         = upright * self._upright_reward_weight
        termination_cost       = termination * self._termination_cost_weight
        ang_vel_xy_cost        = angvel_xy * self._angvel_xy_cost_weight
        pose_reward            = pose * self._pose_reward_weight
        feet_slip_cost         = feet_slip * self._feet_slip_cost_weight
        action_rate_cost       = self._cost_action_rate(action, info["last_act"]) * self._action_rate_cost_weight
        foot_swing_height_cost = foot_swing_height * self._foot_swing_height_cost_weight

        # Update the metrics for rewards
        metrics["reward/tracking_linvel"] = tracking_linvel_reward
        metrics["reward/tracking_angvel"] = tracking_angvel_reward
        metrics["reward/upright"]         = upright_reward
        metrics["cost/ang_vel_xy"]        = ang_vel_xy_cost
        metrics["cost/termination"]       = termination_cost
        metrics["cost/action_rate"]       = action_rate_cost
        metrics["reward/pose"]            = pose_reward
        metrics["cost/feet_slip"]         = feet_slip_cost
        metrics["cost/foot_swing_height"] = foot_swing_height_cost

        total_reward = jp.sum(jp.array([
            tracking_linvel_reward,
            tracking_angvel_reward,
            upright_reward,
            ang_vel_xy_cost,
            termination_cost,
            action_rate_cost,
            feet_slip_cost,
            foot_swing_height_cost,
            pose_reward,
        ]))
        return total_reward


    # All our reward functions used
    def _reward_tracking_linvel(self, data, info) -> jax.Array:
        linvel       = mjx_env.get_sensor_data(self._mj_model, data, "local_linvel")
        command      = info["command"]
        linvel_error = jp.sum(jp.square(command[:2] - linvel[:2]))
        return jp.exp(-linvel_error / 0.25)

    def _reward_tracking_angvel(self, data, info) -> jax.Array:
        angvel       = mjx_env.get_sensor_data(self._mj_model, data, "gyro")
        command      = info["command"]
        angvel_error = jp.square(command[2] - angvel[2])
        return jp.exp(-angvel_error / 0.25)

    def _reward_upright(self, data) -> jax.Array:
        gravity = mjx_env.get_sensor_data(self._mj_model, data, "gravity")
        xy_squared = jp.sum(jp.square(gravity[:2]))
        return jp.exp(-xy_squared / 0.2)

    def _cost_angvel_xy(self, data) -> jax.Array:
        angvel = mjx_env.get_sensor_data(self._mj_model, data, "global_angvel")
        return jp.sum(jp.square(angvel[:2]))

    def _cost_termination(self, done: jax.Array) -> jax.Array:
        return done

    def _reward_pose(self, data, info) -> jax.Array:
        velocity_active_threshold = 0.05
        std_standing = 0.05
        std_moving = 0.35

        joint_positions = data.qpos[self._agent_qpos_ids[7:]]
        error2 = jp.square(joint_positions - self._start_joints)
        command = info["command"]
        speed = jp.linalg.norm(command[:2]) + jp.abs(command[2])
        std = jp.where(
            speed < velocity_active_threshold,
            std_standing,
            std_moving,
        )
        return jp.exp(-jp.mean(error2 / (std ** 2)))

    def _cost_feet_slip(self, data) -> jax.Array:
        contacts = self._get_feet_contacts(data)
        body_linvel = mjx_env.get_sensor_data(self._mj_model, data, "global_linvel")[:2]
        reward = jp.sum(jp.linalg.norm(body_linvel, axis=-1) * contacts)
        return reward

    def _cost_foot_swing_height(self, info: dict[str, Any]) -> jax.Array:
        command_threshold = 0.05
        error = info["landing_peak"] / self._target_height - 1.0
        landing_mask = info["first_contact"].astype(jp.float32)
        cost = jp.sum(jp.square(error) * landing_mask)
        total_command = jp.linalg.norm(info["command"][:2]) + jp.abs(info["command"][2])
        active = (total_command > command_threshold).astype(jp.float32)
        return cost * active

    def _cost_action_rate(self,act: jax.Array, last_act: jax.Array) -> jax.Array:
        return jp.sum(jp.square(act - last_act), axis=-1)

    def _get_feet_contacts(self, data: mjx.Data) -> jax.Array:
        return jp.array([
            mjx_env.get_sensor_data(self._mj_model, data, "left_foot_floor_contact")[0] > 0,
            mjx_env.get_sensor_data(self._mj_model, data, "right_foot_floor_contact")[0] > 0,
        ])

    def _sample_command(self, rng: jax.Array) -> jax.Array:
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

        linvel_x = jax.random.uniform(
            rng1, 
            minval=self._command_range_linvel_x[0], 
            maxval=self._command_range_linvel_x[1]
        )
        linvel_y = jax.random.uniform(
            rng2, 
            minval=self._command_range_linvel_y[0], 
            maxval=self._command_range_linvel_y[1]
        )
        angvel_yaw = jax.random.uniform(
            rng3,
            minval=self._command_range_angvel_yaw[0],
            maxval=self._command_range_angvel_yaw[1],
        )

        # With 10% chance, set everything to zero.
        return jp.where(
            jax.random.bernoulli(rng4, p=0.1),
            jp.zeros(3),
            jp.hstack([linvel_x, linvel_y, angvel_yaw]),
        )

    def _update_feet_height_state(
        self,
        data: mjx.Data,
        info: dict[str, Any],
        contact: jax.Array,
        first_contact: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:

        feet_height = data.site_xpos[self._feet_site_id][:, 2]
        in_air = jp.logical_not(contact)
        swing_peak = jp.where(
            in_air,
            jp.maximum(info["swing_peak"], feet_height),
            info["swing_peak"],
        )

        landing_mask = first_contact.astype(jp.float32)
        landing_peak = swing_peak * landing_mask
        peak_height_sum = info["peak_height_sum"] + jp.sum(landing_peak)
        peak_height_count = info["peak_height_count"] + jp.sum(landing_mask)
        peak_height_mean = jp.where(
            peak_height_count > 0,
            peak_height_sum / peak_height_count,
            0.0,
        )
        peak_height_mean_delta = peak_height_mean - info["peak_height_mean"]

        info["swing_peak"] = jp.where(first_contact, 0.0, swing_peak)
        info["peak_height_sum"] = peak_height_sum
        info["peak_height_count"] = peak_height_count
        info["peak_height_mean"] = peak_height_mean
        return landing_peak, peak_height_mean_delta


    @property
    def observation_size(self):
        return 66

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model

locomotion.register_environment('velocity_zbot', VelocityZbot, default_config)
