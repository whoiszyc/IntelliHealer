from gym.envs.registration import register

register(
    id='FreqControl-v0',
    entry_point='gym_power_res.envs:FreqControlEnv',
)

register(
    id='RestorationDisEnv-v0',
    entry_point='gym_power_res.envs:RestorationDisEnv',
)

register(
    id='RestorationDisEnv-v1',
    entry_point='gym_power_res.envs:RestorationDisEnvRL',
)

register(
    id='RestorationDisEnv119-v0',
    entry_point='gym_power_res.envs:RestorationDisEnv119',
)

register(
    id='RestorationDisVarConEnv-v0',
    entry_point='gym_power_res.envs:RestorationDisVarConEnv',
)

from gym_power_res import envs
from gym_power_res import sorces