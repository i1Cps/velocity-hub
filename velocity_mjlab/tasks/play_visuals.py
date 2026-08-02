"""Shared scene presentation for play environments."""

import mujoco

from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.spec_config import MaterialCfg, TextureCfg


def set_play_atmosphere(spec: mujoco.MjSpec) -> None:
    """Add the soft grey haze and fog used by the Duck Mini Pro viewer."""
    spec.visual.map.haze = 0.3
    spec.visual.map.fogstart = 5.0
    spec.visual.map.fogend = 15.0
    spec.visual.rgba.haze = (0.82, 0.82, 0.82, 1.0)
    spec.visual.rgba.fog = (0.82, 0.82, 0.82, 1.0)


def set_play_terrain_material(terrain: TerrainEntityCfg) -> None:
    """Use the Duck Mini Pro skybox and ground material."""
    terrain.textures = (
        TextureCfg(
            name="skybox",
            type="skybox",
            builtin="gradient",
            rgb1=(0.90, 0.90, 0.90),
            rgb2=(0.82, 0.82, 0.82),
            width=800,
            height=800,
        ),
        TextureCfg(
            name="groundplane",
            type="2d",
            builtin="checker",
            mark="none",
            rgb1=(0.66, 0.68, 0.71),
            rgb2=(0.58, 0.60, 0.63),
            markrgb=(0.58, 0.60, 0.63),
            width=300,
            height=300,
        ),
    )
    terrain.materials = (
        MaterialCfg(
            name="groundplane",
            texture="groundplane",
            texuniform=True,
            texrepeat=(3.0, 3.0),
            reflectance=0.02,
            geom_names_expr=("terrain$",),
        ),
    )
