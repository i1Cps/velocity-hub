import datetime
import functools
import json
import os
import time

# Reduce noisy XLA/CUDA compiler logs.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo
from brax.io import model
import jax
from mujoco_playground import registry
from mujoco_playground import wrapper
import warp as wp

# Register all environments
from main import registry as registry_init

wp.config.quiet = True
logging.set_verbosity(logging.WARNING)

gpu = jax.devices("gpu")[0]
print(f"JAX using GPU: {gpu.device_kind} (platform={gpu.platform}, id={gpu.id})")

# Command Line Arguments:
_ENV_NAME = flags.DEFINE_string(
    "env",
    "duck_mini",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_NUM_TIMESTEPS = flags.DEFINE_integer("num_timesteps", 10_000_000, "Number of timesteps")
_LOG_METRICS   = flags.DEFINE_bool("metrics", True, "Detailed log of metrics per eval")
_IMPL          = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_SEED          = flags.DEFINE_integer("seed", 1, "Random seed")
_SUFFIX        = flags.DEFINE_string("exp_name", None, "Suffix for the experiment name")

def main(argv):
    """Run training and evaluation for the specified environment."""
    del argv

    env_name = "velocity_" + _ENV_NAME.value

    # Load the options
    with open("main/environments/" + env_name + "/options.json") as f:
        options = json.load(f)

    ppo_options = options["ppo"]
    mjx_options = options["mjx"]

    if _NUM_TIMESTEPS.present:
        ppo_options["num_timesteps"]= _NUM_TIMESTEPS.value

    if _IMPL.present:
        mjx_options["impl"]= _IMPL.value

    if _SEED.present:
        ppo_options["seed"] = _SEED.value

    # Load environment configuration
    env_cfg = registry.get_default_config(env_name)
    # Apply sim options
    env_cfg["impl"] = mjx_options["impl"]
    env_cfg["ctrl_dt"] = mjx_options["ctrl_dt"]
    env_cfg["sim_dt"] = mjx_options["sim_dt"]

    env = registry.load(env_name, config=env_cfg)
    metric_dt = float(env.dt)

    print(f"\033[1mmjx options:\033[0m {mjx_options}")
    print(f"\033[1mEnvironment Config:\033[0m\n{env_cfg}")
    def print_ppo_options(options):
        print("\033[1mPPO Training Parameters:\033[0m")
        for key in sorted(options.keys()):
            value = options[key]
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key in sorted(value.keys()):
                    print(f"  {sub_key}: {value[sub_key]}")
            else:
                print(f"{key}: {value}")

    print_ppo_options(ppo_options)
    print("")

    # Generate unique experiment name
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d-%H%M%S")
    exp_name = f"{env_name}-{timestamp}"
    if _SUFFIX.value is not None:
        exp_name += f"-{_SUFFIX.value}"
    print(f"Experiment name: {exp_name}\n")

    # We pass this dict directly to the train function, 
    # so remove network factory as we will configure that manually
    training_params = dict(ppo_options)
    if "network_factory" in training_params:
        del training_params["network_factory"]


    network_fn = ppo_networks.make_ppo_networks
    if "network_factory" in ppo_options:
        network_factory = functools.partial(
            network_fn, **ppo_options["network_factory"]
        )
    else:
        network_factory = network_fn

    train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory = network_factory,
      wrap_env_fn     = wrapper.wrap_for_brax_training,
    )
    times = [time.monotonic()]
    eval_count = 0
    line_width = 80
    colon_center = line_width // 2
    left_width = max(colon_center - 1, 1)
    right_width = max(line_width - colon_center - 1, 1)
    total_evals = ppo_options.get("num_evals")

    # Progress function for logging
    def progress(num_steps, metrics):
        times.append(time.monotonic())
        if not _LOG_METRICS.value:
            print(f"{num_steps}: reward={metrics['eval/episode_reward']:.3f}")
            return

        nonlocal eval_count
        eval_count += 1
        separator = "#" * line_width
        total_str = str(total_evals) if total_evals is not None else "?"
        title = f"Eval {eval_count}/{total_str}"
        print(separator)
        print(f"\033[1m{title.center(line_width)}\033[0m")

        def format_metric_value(value):
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass
            if isinstance(value, float):
                return f"{value:.4f}"
            return str(value)

        def format_metric_line(label, value):
            left = str(label).strip()[:left_width]
            right = f" {format_metric_value(value)}"[:right_width]
            return f"{left:>{left_width}}:{right:<{right_width}}"

        def format_eval_metric_value(key, value):
            if (
                key.startswith("eval/episode_reward/")
                or key.startswith("eval/episode_cost/")
            ):
                return value * metric_dt
            return value

        ordered = [
            ("steps", num_steps),
            ("sps", metrics.get("training/sps")),
            ("reward", metrics.get("eval/episode_reward")),
            ("eval_length", metrics.get("eval/episode_length")),
        ]
        lines = []
        seen = set()
        for key, value in ordered:
            if value is None:
                continue
            line = format_metric_line(key, value)
            if key == "reward":
                line = f"\033[32m{line}\033[0m"
            lines.append(line)
            seen.add(key)

        for key in sorted(metrics.keys()):
            if (
                key in seen
                or key.startswith("training/")
                or key == "eval/walltime"
                or key.endswith("_std")
            ):
                continue
            lines.append(format_metric_line(key, format_eval_metric_value(key, metrics[key])))

        # Keep only top-level eval std metrics and place them near the bottom
        # to avoid clutter from per-component std entries.
        summary_std = [
            ("eval/episode_reward_std", metrics.get("eval/episode_reward_std")),
            ("eval/episode_length_std", metrics.get("eval/episode_length_std")),
        ]
        for key, value in summary_std:
            if value is not None:
                lines.append(format_metric_line(key, value))

        for i, line in enumerate(lines):
            if i == 0:
                print("")
            print(line)

        print("-" * line_width)
        elapsed_seconds = times[-1] - times[0]
        iter_seconds = times[-1] - times[-2] if len(times) > 1 else 0.0
        elapsed_str = time.strftime("%H:%M:%S", time.gmtime(elapsed_seconds))
        print(format_metric_line("Time elapsed", elapsed_str))
        print(format_metric_line("Epoch Time", iter_seconds))


    # Train or load the model
    make_inference_fn, params, training_state = train_fn(
        environment=env,
        progress_fn=progress,
        max_devices_per_host=1
    )

    print("Done training.")
    if len(times) > 1:
        jit_seconds = times[1] - times[0]
        jit_str = time.strftime("%M:%S", time.gmtime(jit_seconds))
        print(f"Time to JIT compile: {jit_str}")
        train_seconds = times[-1] - times[1]
        train_str = time.strftime("%M:%S", time.gmtime(train_seconds))
        print(f"Time to train: {train_str}")


    model_dir_path = os.path.join("models", env_name, "model")
    os.makedirs(os.path.dirname(model_dir_path), exist_ok=True)
    model.save_params(model_dir_path, params)


# For the script tag in pyproject.toml
def cli():
    app.run(main)

if __name__ == "__main__":
  app.run(main)
