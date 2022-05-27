.. _tutorial:

********
Tutorial
********

- IntelliHealer provides imitation learning algorithms with several variations for different feature inputs.
- IntelliHealer can be used as a `Gym <https://gym.openai.com/>`_ environment for distribution system
  restoration to connect with state-of-the-art reinforcement learning algorithms,
  such as `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/?badge=master>`_.
  Currently, it contains two test feeders: 33-node and 119-node system.
- IntelliHealer provides distribution system optimization models built on `Pyomo <http://www.pyomo.org/documentation>`_,
  whicn can be used to develop other problem formulations.


Distribution System Restoration Optimization
=============================================
The distribution system restoration is modeled in the ``OutageManage`` class.

- To solve a restoration problem, we will first to build the problem object as follows:

.. code:: python

    from gym_power_res.envs.DS_pyomo import OutageManage
    problem = OutageManage()

- The ``OutageManage`` will read a system case data specified in ``data_test_case.py``.
  A test case data can be obtained by:

.. code:: python

    import gym_power_res.envs.data_test_case as case
    ppc = case.case33_tieline()

- Then we initialize the problem class with the test case data ``ppc`` and a line outage vector.
  Take a line outage vector ``['line_3', 'line_5', 'line_9']`` as an example.

.. code:: python

    problem.data_preparation(ppc, ['line_3', 'line_5', 'line_9'])

- Here we assume in each time step, only one tie-line can be operated. Then, we need to specify the total time step
  of the problem as follows:

.. code:: python

    problem.initialize_problem(total_time_step)

- Then we can solve the problem using the pre-defined constraints in the function ``solve_network_restoration``.

.. code:: python

    problem.solve_network_restoration()
    opt = pm.SolverFactory("cplex", executable = '/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
    opt.options['mipgap'] = 0
    results = opt.solve(problem.model, tee = True)





Gym for Dsitribution System Restoration
=============================================
The Gym for dsitribution system restoration is built based on the Gym environment template and the optimization models
described in the above section. There are three classes for different scenarios:

- ``RestorationDisEnv``: 33-node Gyn environment for imitation learning
- ``RestorationDisEnvRL``: 33-node Gyn environment for `Stable-Baselines3 <https://stable-baselines3.readthedocs.io/en/master/?badge=master>`_
- ``RestorationDisEnv119``: 119-node Gyn environment for imitation learning


Please refer to the following steps to run the envrironment.

- First, we will import and make the environment with the max/min line outage numbers as follows:

.. code:: python

    from gym_power_res.envs import RestorationDisEnv  # import environment
    ENV_NAME_1 = "RestorationDisEnv-v0"  # define the name of the environment
    env = gym.make(ENV_NAME_1, max_disturbance=2, min_disturbance=2)  # define the environment object

- Second, we will reset the environment. The optional input is a line outage vector,
  such as ``['line_3', 'line_5', 'line_9']``.
  Without the input, the outage will be randomly sampled from all lines.

.. code:: python

    env.reset(['line_6', 'line_11'])

- During the reset process, the restoration optimization problem object named ``sim_case`` is created and initialized. Then we can
  simulate the evolution of the environment under sequetial actions

.. code:: python

    env.step(action_1)
    env.step(action_2)
    env.step(action_3)

- Finally, we can retrieve the results using

.. code:: python

    env.sim_case.get_solution_2d('bus_variable_name', env.sim_case.iter_bus, env.sim_case.iter_time)
    env.sim_case.get_solution_2d('line_variable_name', env.sim_case.iter_line, env.sim_case.iter_time)


Imitation Learning
=============================================
The imitation learning algoritm is implemented in the function ``main_behavior_cloning``. It requires the ``env`` object
and the ``agent`` object. The operation of the alforithm is described below with self-explanatory comments.

.. code:: python

    def main_behavior_cloning(output_path):
    """ BC algorithm
    """
    # ============= create GYM environment ===============
    env = gym.make(ENV_NAME_1, , max_disturbance=1, min_disturbance=1)

    # ============== create agent ===================
    agent = Agent(env, output_path)

    # ============= Begin main training loop ===========
    flag_convergence = False   # set convergence flag to be false
    tic = time.perf_counter()  # start clock
    for it in range(NUM_TOTAL_EPISODES):
        if it % 1 == 0:
            toc = time.perf_counter()
            print("===================================================")
            print(f"Training time: {toc - tic:0.4f} seconds; Mission {it:d} of {NUM_TOTAL_EPISODES:d}")
            print("===================================================")
        agent.logger.info(f"=============== Mission {it:d} of {NUM_TOTAL_EPISODES:d} =================")

        # initialize environment
        s0, l0 = env.reset()

        # get expert policy
        # note that expert only retrieve information from the environment but will not change it
        agent.get_expert_policy(env, s0)

        # determine if use expert advice or learned policy
        # main difference between behavior cloning and DAGGER
        if it > WARM_START_OFF:
            agent.warmstart = False

        # executes the expert policy and perform imitation learning
        agent.run_train(env, s0, l0)

        # test current trained policy network using new environment from certain iterations
        if it >= 0:
            # initialize environment
            s0, l0 = env.reset()
            # execute learned policy on the environment
            flag_convergence = agent.run_test(env, s0)

        if flag_convergence == True:
            break

        agent.total_episode = agent.total_episode + 1

    return agent








