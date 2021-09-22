import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


register(
    id='panda-v0',
    entry_point='gym_panda.envs:PandaEnv',
)

register(
    id='disabledpanda-v0',
    entry_point='gym_panda.envs:DisabledPandaEnv',
)

register(
    id='feasibilitypanda-v0',
    entry_point='gym_panda.envs:FeasibilityPandaEnv',
)

register(
    id='realpanda-v0',
    entry_point='gym_panda.envs:RealPandaEnv',
)