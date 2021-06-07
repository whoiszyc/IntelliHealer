
*****************
Main Functions
*****************

There are three major classes in ``IntelliHealer``:

- ``OutageManage``: formulate and solve restoration optimization problems
- ``RestorationDisEnv``, ``RestorationDisEnv119``, ``RestorationDisEnvRL``: interact with agent for learning
- ``Agent``: solve imitation and reinforcement learning problems


OutageManage
=============================================

.. autoclass:: gym_power_res.envs.DS_pyomo.OutageManage
    :members:
    :undoc-members:
    :show-inheritance:

Restoration Gym
===================

.. autoclass:: gym_power_res.envs.GYM_ENV_restoration_distribution.RestorationDisEnvRL
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: gym_power_res.envs.GYM_ENV_restoration_distribution.RestorationDisEnv
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: gym_power_res.envs.GYM_ENV_restoration_distribution.RestorationDisEnv119
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: gym_power_res.envs.GYM_ENV_restoration_distribution.RestorationDisVarConEnv
    :members:
    :undoc-members:
    :show-inheritance:

Restoration Agent
===================

.. autoclass:: project_dis_restoration.train_IL_BC_tieline.Agent
    :members:
    :undoc-members:
    :show-inheritance:
