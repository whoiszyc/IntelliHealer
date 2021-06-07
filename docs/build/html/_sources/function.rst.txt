.. _function:

*****************
Main Functions
*****************

This chapter contains advanced topics on modeling and simulation and how they are implemented in ANDES.
It aims to provide an in-depth explanation of how the ANDES framework is set up for symbolic modeling and
numerical simulation. It also provides an example for interested users to implement customized DAE models.

System
=======

Overview
--------
System is the top-level class for organizing power system models and orchestrating calculations.

.. autoclass:: gym_power_res.envs.GYM_ENV_restoration_distribution.RestorationDisEnvRL
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: gym_power_res.envs.DS_pyomo.OutageManage
    :members:
    :undoc-members:
    :show-inheritance:

.. note::
    `andes.System` is an alias of `andes.system.System`.


Dynamic Imports

System dynamically imports groups, models, and routines at creation.
To add new models, groups or routines, edit the corresponding file by adding entries following examples.

.. autofunction:: project_dis_restoration.train_IL_BC_tieline.main_behavior_cloning
    :noindex:


Code Generation

Under the hood, all symbolically defined equations need to be generated into anonymous function calls for
accelerating numerical simulations.
This process is automatically invoked for the first time ANDES is run command line.
It takes several seconds up to a minute to finish the generation.

.. note::
    Code generation has been done if one has executed ``andes``, ``andes selftest``, or ``andes prepare``.

.. warning::
    When models are modified (such as adding new models or changing equation strings), code generation needs
    to be executed again for consistency. It can be more conveniently triggered from command line with
    ``andes prepare -i``.
