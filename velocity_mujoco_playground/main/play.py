import time
from absl import app
from absl import flags

import jax
import mujoco
import mujoco.viewer

from .registry import PLAY_ENV_REGISTRY

# Command Line Arguments:
_ENV_NAME = flags.DEFINE_string(
    "env",
    "zbot",
    "pick an environment name"
)


def _set_camera_auto_track(viewer, model):
    for body_id in range(model.nbody):
        is_weld = model.body_weldid[body_id] == 0
        root_is_mocap = model.body_mocapid[model.body_rootid[body_id]] >= 0
        if not (is_weld and not root_is_mocap):
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = body_id
            viewer.cam.fixedcamid = -1
            break


# Play environments
def main(argv):
    if _ENV_NAME.value not in PLAY_ENV_REGISTRY:
        raise ValueError(f"Unknown environment '{_ENV_NAME.value}'.")

    # Create JAX and Numpy RNG
    rng = jax.random.PRNGKey(0)

    # Instantiate the play environment
    play_env = PLAY_ENV_REGISTRY[_ENV_NAME.value]()

    model = play_env._mj_model
    data  = mujoco.MjData(model)

    # Reset the environment
    data, rng = play_env.reset(data, rng)
    mujoco.mj_forward(model, data)

    paused = False

    def key_callback(keycode):
        nonlocal paused, rng, data

        if keycode == ord(" "):
            paused = not paused

        elif keycode == ord("R"):
            data, rng = play_env.reset(data, rng)
            mujoco.mj_forward(model, data)
            viewer.sync()

        # Delegate to environment-specific key handler
        elif hasattr(play_env, "key_callback"):
            play_env.key_callback(keycode)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.sync()

        # Enable fog rendering
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_FOG] = 1

        with viewer.lock():
            _set_camera_auto_track(viewer, model)
            viewer.cam.distance = 3.0
            viewer.cam.elevation = -5.0
            viewer.cam.azimuth = 90.0

        while viewer.is_running():
            viewer.sync()
            if not paused:
                t0 = time.perf_counter()
                model, data, rng = play_env.step(model, data, rng)

                mujoco.mj_forward(model, data)
                viewer.sync()

                elapsed = time.perf_counter() - t0
                to_sleep = play_env.ctrl_dt - elapsed
                if to_sleep > 0:
                    time.sleep(to_sleep)


def cli():
    app.run(main)


if __name__ == "__main__":
    app.run(main)
