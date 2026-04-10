# Training environments
from main.environments.velocity_humanoid.env import VelocityHumanoid
from main.environments.velocity_quadruped.env import VelocityQuadruped
from main.environments.velocity_kbot.env import VelocityKbot
from main.environments.velocity_zbot.env import VelocityZbot
from main.environments.velocity_h1.env import VelocityH1
from main.environments.velocity_t1.env import VelocityT1
from main.environments.velocity_anymal_c.env import VelocityAnymalC
from main.environments.velocity_spot.env import VelocitySpot
from main.environments.velocity_duck_mini.env import VelocityDuckMini

# Play environments
from main.environments.velocity_humanoid.play import VelocityHumanoid_Play
from main.environments.velocity_quadruped.play import VelocityQuadruped_Play
from main.environments.velocity_kbot.play import VelocityKbot_Play
from main.environments.velocity_zbot.play import VelocityZbot_Play
from main.environments.velocity_t1.play import VelocityT1_Play
from main.environments.velocity_h1.play import VelocityH1_Play
from main.environments.velocity_anymal_c.play import VelocityAnymalC_Play
from main.environments.velocity_spot.play import VelocitySpot_Play
from main.environments.velocity_duck_mini.play import VelocityDuckMini_Play

PLAY_ENV_REGISTRY = {
    "humanoid": VelocityHumanoid_Play,
    "quadruped": VelocityQuadruped_Play,
    "kbot": VelocityKbot_Play,
    "zbot": VelocityZbot_Play,
    "h1": VelocityH1_Play,
    "t1": VelocityT1_Play,
    "anymal_c": VelocityAnymalC_Play,
    "spot": VelocitySpot_Play,
    "duck_mini": VelocityDuckMini_Play,
}

