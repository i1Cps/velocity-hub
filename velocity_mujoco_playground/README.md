<h1 align="center">🛝 Velocity MuJoCo Playground</h1>

<p align="center">The MuJoCo Playground side of Velocity Hub is built for fast RL research. Train bipeds and quadrupeds to follow velocity commands in minutes with simple reward shaping, and clear robot specific observation design without long feedback loops.</p>

## Environments

<div align="center">

| anymal_c | duck_mini | h1 | humanoid |
| --- | --- | --- | --- |
| <img src="../media/anymal_c_square.png" width="100"/> | <img src="../media/duck_mini_square.png" width="100"/> | <img src="../media/h1_square.png" width="100"/> | <img src="../media/humanoid_square.png" width="100"/> |

| kbot | quadruped | spot | t1 | zbot |
| --- | --- | --- | --- | --- |
| <img src="../media/kbot_square.png" width="100"/> | <img src="../media/quadruped_square.png" width="100"/> | <img src="../media/spot_square.png" width="100"/> | <img src="../media/t1_square.png" width="100"/> | <img src="../media/zbot_square.png" width="100"/> |

</div>

## Training Command Examples

```bash
uv run train --env=duck_mini
```
```bash
uv run train --env=h1
```
```bash
uv run train --env=kbot
```
```bash
uv run train --env=spot
```

Trained models are saved to `models/velocity_<env>/model`. If wandb is configured on your system, runs are automatically logged to your account.

## Running a Policy

```bash
uv run play --env=kbot
```

The play script opens an interactive viewer and loads the model from `models/velocity_<env>/model`. Use keyboard controls to issue velocity commands:

| Key | Command |
| --- | --- |
| UP / DOWN | lin_vel_x +/- |
| J / L | lin_vel_y +/- |
| LEFT / RIGHT | ang_vel_z +/- |
| X | zero all commands |
| SPACE | pause / resume |
| R | reset environment |
