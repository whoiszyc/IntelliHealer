# Import stuff from gym
import gym
from gym import error, spaces, utils
from gym.utils import seeding


# Import stuff from gym power restoration
from gym_power_res.envs.DS_pyomo import OutageManage
from gym_power_res.envs.data_test_case import case33_tieline, case33_tieline_DG, case119_tieline
import pyomo.environ as pm
from pyomo.opt import SolverStatus, TerminationCondition

# Import general modules
import sys
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import sympy as sy
import pandas as pd
import networkx as nx
from collections import OrderedDict
from sys import exit


class RestorationDisEnvRL(gym.Env):
  """
  Restoration env for StableBaseline3 (reinforcement learning algorithm collections)
  """
  
  metadata = {'render.modes': ['human']}

  def __init__(self, max_disturbance, min_disturbance):
    """
    initialize network data and status
    """
    # network data
    self.ppc = case33_tieline()
    num_line = len(self.ppc['iter_line'])
    num_bus = len(self.ppc['iter_bus'])
    num_tieline = len(self.ppc['tieline'])

    # exploration number is equal to tie-line numbers
    # consider that
    # (1) range(5) = [0, 1, 2, 3, 4, 5];
    # (2) in optimization we model initial step and enforce all tielines to be zero
    # we set exploration_total to be five
    self.exploration_total = num_tieline
    self.exploration_seq_idx = [i for i in range(self.exploration_total + 1)]

    # action number is equal to the tie-line plus one (do nothing option)
    self.action_space = spaces.Discrete(num_tieline + 1)
    self.observation_space = spaces.Box(np.array([0] * (num_line)), np.array([1] * (num_line)), dtype=np.int)

    # =========== disturbance upper bound =============
    self.max_disturbance = max_disturbance
    self.min_disturbance = min_disturbance

    # ============ voltage specifications ============
    self.VS = 1.05
    self.dV = 0.05

    self.seed()
    self.viewer = None
    self.state = None

    self.steps_beyond_done = None


  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]



  def reset(self, disturbance=None):
    """
    Here we should setup the episode control for the total time horizon, which will be independent of individual segment simulation
    """
    # ================= define system states for a new episode ===============
    # ordered dict is very important since learning data is in the form of matrices
    self.state_line_status = OrderedDict()
    self.state_load_status = OrderedDict()

    # initialize a list to store the load status during an episode
    self.load_value_current = 0
    self.load_value_episode = []

    # index to determine the instants
    self.exploration_index = 0
    print('Exploration index is reset.')

    # ================== generate disturbance ====================
    # generate random disturbance if no specific one is given
    if disturbance == None:
      # define the disturbance set
      disturbance_set = ['line_3', 'line_4', "line_5", 'line_6', 'line_7', 'line_8', 'line_9', 'line_10', 'line_11', 'line_12', 'line_13', 'line_14',
                         'line_15', 'line_16', 'line_17', 'line_18', 'line_19', 'line_20', 'line_21', 'line_22', 'line_23', 'line_24', 'line_25',
                         'line_26', 'line_27', 'line_28', 'line_29', 'line_30', 'line_31', 'line_32']
      # disturbance_set = ['line_11', 'line_22', 'line_9']
      # generate disturbance upper bound for this episoid
      num_disturbance = random.randint(self.min_disturbance, self.max_disturbance)
      # record generated disturbance
      self.disturbance = []
      for i in range(num_disturbance):
        # generate one line outage at a time
        random_disturbance = random.choice(disturbance_set)
        # record
        self.disturbance.append(random_disturbance)
        # remove from the set
        disturbance_set.remove(random_disturbance)
    else:
      self.disturbance = disturbance

    # =============== initialize the line for optimization ===============
    # initialize line status
    for i in self.ppc['iter_line']:
      self.state_line_status[i] = 1
    for i in self.ppc['tieline']:
      self.state_line_status[i] = 0   # initially controls are zero
      # update line status
    for i in self.disturbance:
      self.state_line_status[i] = 0

    # =============== solve the load status given line status  ===============
    self.sim_case = OutageManage()
    self.sim_case.data_preparation(self.ppc, self.disturbance, self.VS, self.dV)
    self.sim_case.solve_load_pickup(self.state_line_status)
    opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
    # opt = pm.SolverFactory("cplex")
    opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
    results = opt.solve(self.sim_case.model, tee=False)

    # check feasibility
    # ------------------------- if optimal, update system status -------------------
#     if results['solver'][0]['Termination condition'].key == 'optimal':
    if results.solver.termination_condition == TerminationCondition.optimal:
      # ................get optimization results................
      sol_load_status = self.sim_case.get_solution_2d('rho', self.sim_case.iter_bus, self.sim_case.iter_time)
      sol_line_status = self.sim_case.get_solution_2d('ul', self.sim_case.iter_line, self.sim_case.iter_time)
      # .................calculate current total load............
      _temp_load_t = []
      for t in self.sim_case.iter_time:
        _temp_load = []
        for i in self.sim_case.iter_bus:
          id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
          idx = np.where(id == self.ppc['bus'][:, 0])[0][0]
          _temp_load.append(sol_load_status[i][t] * self.ppc['bus'][idx, 4])
        _temp_load_t.append(sum(_temp_load))
      # update the current load value
      self.load_value_current = sum(_temp_load_t)
      # record
      self.load_value_episode.append(self.load_value_current)

      # .............. verify if solutions are the same with the given conditions ..............
      for i in self.state_line_status.keys():
        if abs(self.state_line_status[i] - sol_line_status[i][0]) > 1e-6:
          print('GYM_Power: solutions and conditions are not the same for ' + i)

      #.............. update system status and observation .............
      for i in sol_load_status.keys():
        self.state_load_status[i] = sol_load_status[i][0]

      # ............. initialize observations................
      self.current_observ = []
      for i in self.state_line_status.keys():
        self.current_observ.append(self.state_line_status[i])

    # ------------------------- if infeasible TODO: maybe use try structure -------------------
    else:
      print("GYM_Power: Current generated disturbance makes the system infeasible")
      print("GYM_Power: Disturbance is: ", self.disturbance)
      sys.exit()

    # convert load status to a list
    _load_status = self.dict2list(self.state_load_status)

    return self.current_observ


  def step(self, action):
    """
    Apply the given actions to the environment for one step
    """
    # check is action format is correct
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    # ===================== parse action from GYM format to optimization format =====================
    # get current line status; remember to use copy so that modification of "action_line" will not impact "self.state_line_status"
    action_line = self.state_line_status.copy()
    if action == 1:
      action_line['line_33'] = int(not (action_line['line_33']))
    elif action == 2:
      action_line['line_34'] = int(not (action_line['line_34']))
    elif action == 3:
      action_line['line_35'] = int(not (action_line['line_35']))
    elif action == 4:
      action_line['line_36'] = int(not (action_line['line_36']))
    elif action == 5:
      action_line['line_37'] = int(not (action_line['line_37']))
    elif action == 0:
      pass
    else:
      print("GYM_Power: Out of action space")


    # ===================== determine termination condition and rewards =====================
    # first check if this is the last step in this episode
    if self.exploration_index == self.exploration_total:
      done = True  # we are done with this episode
      reward = 0
    else:
      done = False

      # =====================  solve for load status =====================
      self.sim_case.solve_load_pickup(action_line, self.state_load_status)
      opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
      # opt = pm.SolverFactory("cplex")
      opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
      results = opt.solve(self.sim_case.model, tee=False)

      # ----------- if infeasible, do nothing -------------
#       if results['solver'][0]['Termination condition'].key == 'infeasible':
      if results.solver.termination_condition == TerminationCondition.infeasible:
        reward = -1000
        self.solver_condition = 'infeasible'
        # if infeasible, we do not update the system status
        # add the current load status into the episode list
        self.load_value_episode.append(self.load_value_current)

      # --------------- if optimal, update status, observations and determine rewards ----------------
#       elif results['solver'][0]['Termination condition'].key == 'optimal':
      elif results.solver.termination_condition == TerminationCondition.optimal:
        self.solver_condition = 'optimal'
        # if optimal, get results
        sol_load_status = self.sim_case.get_solution_2d('rho', self.sim_case.iter_bus, self.sim_case.iter_time)
        sol_line_status = self.sim_case.get_solution_2d('ul', self.sim_case.iter_line, self.sim_case.iter_time)

        # ...............verify if solutions are the same with the given conditions............
        for i in action_line.keys():
          if abs(action_line[i] - sol_line_status[i][0]) > 1e-6:
            print('GYM_Power: solutions and conditions are not the same for ' + i)

        # ................. update system status and observation .................
        # retrieve bus-indexed status
        for i in sol_load_status.keys():
          self.state_load_status[i] = sol_load_status[i][0]
        # retrieve line-indexed status
        for i in sol_line_status.keys():
          self.state_line_status[i] = sol_line_status[i][0]

          # fresh observations
          self.current_observ = []
          for i in self.state_line_status.keys():
            self.current_observ.append(self.state_line_status[i])

        # ................. get new total load value and determine rewards .................
        _temp_load_t = []
        for t in self.sim_case.iter_time:
          _temp_load = 0
          for i in self.sim_case.iter_bus:
            id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
            idx = np.where(id == self.ppc['bus'][:, 0])[0][0]
            _temp_load = _temp_load + sol_load_status[i][t] * self.ppc['bus'][idx, 4]
          _temp_load_t.append(_temp_load)
        load_value_new = sum(_temp_load_t)
        # compare the current and new load value to determine the rewards
        if load_value_new > self.load_value_current:
          reward = 150
        elif load_value_new == self.load_value_current:
          reward = -10
        elif load_value_new < self.load_value_current:
          reward = -100
        else:
          print('GYM_Power: impossible load condition')
          sys.exit()
        # update load value and append it into the episode list
        self.load_value_current = load_value_new
        self.load_value_episode.append(self.load_value_current)

      else:
        print('GYM_Power: Unknown solver condition')
        sys.exit()

      # update index
      self.exploration_index += 1

    # convert load status to a list
    _load_status = self.dict2list(self.state_load_status)

    return self.current_observ, reward, done, {}


  def dict2list(self, dict):
    """
    convert an order dict to a list
    """
    a = []
    for k in dict.keys():
      a.append(dict[k])
    return a


  def render(self, mode='human'):
    pass


  def close(self):
    pass






class RestorationDisEnv(gym.Env):
  """
  Restoration env for imitation learning
  """

  metadata = {'render.modes': ['human']}

  def __init__(self, max_disturbance, min_disturbance):
    # ================== initialize network data and status ====================
    # network data
    self.ppc = case33_tieline()
    num_line = len(self.ppc['iter_line'])
    num_bus = len(self.ppc['iter_bus'])
    num_tieline = len(self.ppc['tieline'])

    # exploration number is equal to tie-line numbers
    # consider that
    # (1) range(5) = [0, 1, 2, 3, 4, 5];
    # (2) in optimization we model initial step and enforce all tielines to be zero
    # we set exploration_total to be five
    self.exploration_total = num_tieline
    self.exploration_seq_idx = [i for i in range(self.exploration_total + 1)]

    # action number is equal to the tie-line plus one (do nothing option)
    self.action_space = spaces.Discrete(num_tieline + 1)
    self.observation_space = spaces.Box(np.array([0] * (num_line)), np.array([1] * (num_line)), dtype=np.int)

    # =========== disturbance upper bound =============
    self.max_disturbance = max_disturbance
    self.min_disturbance = min_disturbance

    # ============ voltage specifications ============
    self.VS = 1.05
    self.dV = 0.05

    self.seed()
    self.viewer = None
    self.state = None

    self.steps_beyond_done = None


  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]



  def reset(self, disturbance=None):
    """
    Here we should setup the episode control for the total time horizon, which will be independent of individual segment simulation
    """
    # ================= define system states for a new episode ===============
    # ordered dict is very important since learning data is in the form of matrices
    self.state_line_status = OrderedDict()
    self.state_load_status = OrderedDict()

    # initialize a list to store the load status during an episode
    self.load_value_current = 0
    self.load_value_episode = []

    # index to determine the instants
    self.exploration_index = 0
    # print('Exploration index is reset to one.')

    # ================== generate disturbance ====================
    # generate random disturbance if no specific one is given
    if disturbance == None:
      # define the disturbance set
      disturbance_set = ['line_3', 'line_4', "line_5", 'line_6', 'line_7', 'line_8', 'line_9', 'line_10', 'line_11', 'line_12', 'line_13', 'line_14',
                         'line_15', 'line_16', 'line_17', 'line_18', 'line_19', 'line_20', 'line_21', 'line_22', 'line_23', 'line_24', 'line_25',
                         'line_26', 'line_27', 'line_28', 'line_29', 'line_30', 'line_31', 'line_32']
      # disturbance_set = ['line_11', 'line_22', 'line_9']
      # generate disturbance upper bound for this episoid
      num_disturbance = random.randint(self.min_disturbance, self.max_disturbance)
      # record generated disturbance
      self.disturbance = []
      for i in range(num_disturbance):
        # generate one line outage at a time
        random_disturbance = random.choice(disturbance_set)
        # record
        self.disturbance.append(random_disturbance)
        # remove from the set
        disturbance_set.remove(random_disturbance)
    else:
      self.disturbance = disturbance

    # =============== initialize the line for optimization ===============
    # initialize line status
    for i in self.ppc['iter_line']:
      self.state_line_status[i] = 1
    for i in self.ppc['tieline']:
      self.state_line_status[i] = 0   # initially controls are zero
      # update line status
    for i in self.disturbance:
      self.state_line_status[i] = 0

    # =============== solve the load status given line status  ===============
    self.sim_case = OutageManage()
    self.sim_case.data_preparation(self.ppc, self.disturbance, self.VS, self.dV)
    self.sim_case.solve_load_pickup(self.state_line_status)
    opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
    # opt = pm.SolverFactory("cplex")
    opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
    results = opt.solve(self.sim_case.model, tee=False)

    # check feasibility
    # ------------------------- if optimal, update system status -------------------
#     if results['solver'][0]['Termination condition'].key == 'optimal':
    if results.solver.termination_condition == TerminationCondition.optimal:
      # ................get optimization results................
      sol_load_status = self.sim_case.get_solution_2d('rho', self.sim_case.iter_bus, self.sim_case.iter_time)
      sol_line_status = self.sim_case.get_solution_2d('ul', self.sim_case.iter_line, self.sim_case.iter_time)
      # .................calculate current total load............
      _temp_load_t = []
      for t in self.sim_case.iter_time:
        _temp_load = []
        for i in self.sim_case.iter_bus:
          id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
          idx = np.where(id == self.ppc['bus'][:, 0])[0][0]
          _temp_load.append(sol_load_status[i][t] * self.ppc['bus'][idx, 4])
        _temp_load_t.append(sum(_temp_load))
      # update the current load value
      self.load_value_current = sum(_temp_load_t)
      # record
      self.load_value_episode.append(self.load_value_current)

      # .............. verify if solutions are the same with the given conditions ..............
      for i in self.state_line_status.keys():
        if abs(self.state_line_status[i] - sol_line_status[i][0]) > 1e-6:
          print('GYM_Power: solutions and conditions are not the same for ' + i)

      #.............. update system status and observation .............
      for i in sol_load_status.keys():
        self.state_load_status[i] = sol_load_status[i][0]

      # ............. initialize observations................
      self.current_observ = []
      for i in self.state_line_status.keys():
        self.current_observ.append(self.state_line_status[i])

    # ------------------------- if infeasible TODO: maybe use try structure -------------------
    else:
      print("GYM_Power: Current generated disturbance makes the system infeasible")
      print("GYM_Power: Disturbance is: ", self.disturbance)
      sys.exit()

    # convert load status to a list
    _load_status = self.dict2list(self.state_load_status)

    return np.array(self.current_observ), np.array(_load_status)


  def step(self, action):
    """
    Apply the given actions to the environment for one step
    """
    # check is action format is correct
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    # ===================== parse action from GYM format to optimization format =====================
    # get current line status; remember to use copy so that modification of "action_line" will not impact "self.state_line_status"
    action_line = self.state_line_status.copy()
    if action == 1:
      action_line['line_33'] = int(not (action_line['line_33']))
    elif action == 2:
      action_line['line_34'] = int(not (action_line['line_34']))
    elif action == 3:
      action_line['line_35'] = int(not (action_line['line_35']))
    elif action == 4:
      action_line['line_36'] = int(not (action_line['line_36']))
    elif action == 5:
      action_line['line_37'] = int(not (action_line['line_37']))
    elif action == 0:
      pass
    else:
      print("GYM_Power: Out of action space")


    # ===================== determine termination condition and rewards =====================
    # first check if this is the last step in this episode
    if self.exploration_index == self.exploration_total:
      done = True  # we are done with this episode
      reward = 0
    else:
      done = False

      # =====================  solve for load status =====================
      self.sim_case.solve_load_pickup(action_line, self.state_load_status)
      opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
      # opt = pm.SolverFactory("cplex")
      opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
      results = opt.solve(self.sim_case.model, tee=False)

      # ----------- if infeasible, do nothing -------------
#       if results['solver'][0]['Termination condition'].key == 'infeasible':
      if results.solver.termination_condition == TerminationCondition.infeasible:
        reward = -1000
        self.solver_condition = 'infeasible'
        # if infeasible, we do not update the system status
        # add the current load status into the episode list
        self.load_value_episode.append(self.load_value_current)

      # --------------- if optimal, update status, observations and determine rewards ----------------
#       elif results['solver'][0]['Termination condition'].key == 'optimal':
      elif results.solver.termination_condition == TerminationCondition.optimal:
        self.solver_condition = 'optimal'
        # if optimal, get results
        sol_load_status = self.sim_case.get_solution_2d('rho', self.sim_case.iter_bus, self.sim_case.iter_time)
        sol_line_status = self.sim_case.get_solution_2d('ul', self.sim_case.iter_line, self.sim_case.iter_time)

        # ...............verify if solutions are the same with the given conditions............
        for i in action_line.keys():
          if abs(action_line[i] - sol_line_status[i][0]) > 1e-6:
            print('GYM_Power: solutions and conditions are not the same for ' + i)

        # ................. update system status and observation .................
        # retrieve bus-indexed status
        for i in sol_load_status.keys():
          self.state_load_status[i] = sol_load_status[i][0]
        # retrieve line-indexed status
        for i in sol_line_status.keys():
          self.state_line_status[i] = sol_line_status[i][0]

          # fresh observations
          self.current_observ = []
          for i in self.state_line_status.keys():
            self.current_observ.append(self.state_line_status[i])

        # ................. get new total load value and determine rewards .................
        _temp_load_t = []
        for t in self.sim_case.iter_time:
          _temp_load = 0
          for i in self.sim_case.iter_bus:
            id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
            idx = np.where(id == self.ppc['bus'][:, 0])[0][0]
            _temp_load = _temp_load + sol_load_status[i][t] * self.ppc['bus'][idx, 4]
          _temp_load_t.append(_temp_load)
        load_value_new = sum(_temp_load_t)
        # compare the current and new load value to determine the rewards
        if load_value_new > self.load_value_current:
          reward = 150
        elif load_value_new == self.load_value_current:
          reward = -10
        elif load_value_new < self.load_value_current:
          reward = -100
        else:
          print('GYM_Power: impossible load condition')
          sys.exit()
        # update load value and append it into the episode list
        self.load_value_current = load_value_new
        self.load_value_episode.append(self.load_value_current)

      else:
        print('GYM_Power: Unknown solver condition')
        sys.exit()

      # update index
      self.exploration_index += 1

    # convert load status to a list
    _load_status = self.dict2list(self.state_load_status)

    return np.array(self.current_observ), reward, done, np.array(_load_status)


  def dict2list(self, dict):
    """
    convert an order dict to a list
    """
    a = []
    for k in dict.keys():
      a.append(dict[k])
    return a


  def render(self, mode='human'):
    pass


  def close(self):
    pass






class RestorationDisEnv119(gym.Env):
  """
  Restoration env for imitation learning
  """

  metadata = {'render.modes': ['human']}

  def __init__(self, max_disturbance, min_disturbance):
    # ================== initialize network data and status ====================
    # network data
    self.ppc = case119_tieline()
    num_line = len(self.ppc['iter_line'])
    num_bus = len(self.ppc['iter_bus'])
    num_tieline = len(self.ppc['tieline'])

    # exploration number is equal to tie-line numbers
    # consider that
    # (1) range(5) = [0, 1, 2, 3, 4, 5];
    # (2) in optimization we model initial step and enforce all tielines to be zero
    # we set exploration_total to be five
    self.exploration_total = num_tieline
    self.exploration_seq_idx = [i for i in range(self.exploration_total + 1)]

    # action number is equal to the tie-line plus one (do nothing option)
    self.action_space = spaces.Discrete(num_tieline + 1)
    self.observation_space = spaces.Box(np.array([0] * (num_line)), np.array([1] * (num_line)), dtype=np.int)

    # =========== disturbance upper bound =============
    self.max_disturbance = max_disturbance
    self.min_disturbance = min_disturbance

    # ============ voltage specifications ============
    self.VS = 1.05
    self.dV = 0.05

    self.seed()
    self.viewer = None
    self.state = None

    self.steps_beyond_done = None


  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]



  def reset(self, disturbance=None):
    """
    Here we should setup the episode control for the total time horizon, which will be independent of individual segment simulation
    """
    # ================= define system states for a new episode ===============
    # ordered dict is very important since learning data is in the form of matrices
    self.state_line_status = OrderedDict()
    self.state_load_status = OrderedDict()

    # initialize a list to store the load status during an episode
    self.load_value_current = 0
    self.load_value_episode = []

    # index to determine the instants
    self.exploration_index = 0
    # print('Exploration index is reset to one.')

    # ================== generate disturbance ====================
    # generate random disturbance if no specific one is given
    if disturbance == None:
      # define the disturbance set
      disturbance_set = ['line_1', 'line_2', 'line_3', 'line_4', "line_5", 'line_6', 'line_7', 'line_8', 'line_9',
                         'line_10', 'line_11', 'line_12', 'line_13', 'line_14',
                         'line_15', 'line_16', 'line_17', 'line_18', 'line_19', 'line_20', 'line_21', 'line_22',
                         'line_23', 'line_24', 'line_25',
                         'line_26', 'line_27', 'line_28', 'line_29', 'line_30', 'line_31', 'line_32', 'line_33',
                         'line_34', 'line_35', 'line_36',
                         'line_37', 'line_38', 'line_39', 'line_40', 'line_41', 'line_42', 'line_43', 'line_44',
                         'line_45', 'line_46', 'line_47',
                         'line_48', 'line_49', 'line_50', 'line_51', 'line_52', 'line_53', 'line_54', 'line_55',
                         'line_56', 'line_57', 'line_58',
                         'line_59', 'line_60', 'line_61', 'line_62', 'line_63', 'line_64', 'line_65', 'line_66',
                         'line_67', 'line_68', 'line_69',
                         'line_70', 'line_71', 'line_72', 'line_73', 'line_74', 'line_75', 'line_76', 'line_77',
                         'line_78', 'line_79', 'line_80','line_81', 'line_82', 'line_83', 'line_84', 'line_85',
                         'line_86', 'line_87', 'line_88', 'line_89', 'line_90', 'line_91', 'line_92', 'line_93',
                         'line_94', 'line_95', 'line_96', 'line_97', 'line_98', 'line_99', 'line_100', 'line_101',
                         'line_102', 'line_103', 'line_104', 'line_105', 'line_106', 'line_107', 'line_108', 'line_109',
                         'line_110', 'line_111', 'line_112', 'line_113', 'line_114', 'line_115', 'line_116', 'line_117',
                         ]
      # generate disturbance upper bound for this episoid
      num_disturbance = random.randint(self.min_disturbance, self.max_disturbance)
      # record generated disturbance
      self.disturbance = []
      for i in range(num_disturbance):
        # generate one line outage at a time
        random_disturbance = random.choice(disturbance_set)
        # record
        self.disturbance.append(random_disturbance)
        # remove from the set
        disturbance_set.remove(random_disturbance)
    else:
      self.disturbance = disturbance

    # =============== initialize the line for optimization ===============
    # initialize line status
    for i in self.ppc['iter_line']:
      self.state_line_status[i] = 1
    for i in self.ppc['tieline']:
      self.state_line_status[i] = 0   # initially controls are zero
      # update line status
    for i in self.disturbance:
      self.state_line_status[i] = 0

    # =============== solve the load status given line status  ===============
    self.sim_case = OutageManage()
    self.sim_case.data_preparation(self.ppc, self.disturbance, self.VS, self.dV)
    self.sim_case.solve_load_pickup(self.state_line_status)
    opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')  #
    # opt = pm.SolverFactory("cplex")
    opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
    results = opt.solve(self.sim_case.model, tee=False)

    # check feasibility
    # ------------------------- if optimal, update system status -------------------
#     if results['solver'][0]['Termination condition'].key == 'optimal':
    if results.solver.termination_condition == TerminationCondition.optimal:
      # ................get optimization results................
      sol_load_status = self.sim_case.get_solution_2d('rho', self.sim_case.iter_bus, self.sim_case.iter_time)
      sol_line_status = self.sim_case.get_solution_2d('ul', self.sim_case.iter_line, self.sim_case.iter_time)
      # .................calculate current total load............
      _temp_load_t = []
      for t in self.sim_case.iter_time:
        _temp_load = []
        for i in self.sim_case.iter_bus:
          id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
          idx = np.where(id == self.ppc['bus'][:, 0])[0][0]
          _temp_load.append(sol_load_status[i][t] * self.ppc['bus'][idx, 4])
        _temp_load_t.append(sum(_temp_load))
      # update the current load value
      self.load_value_current = sum(_temp_load_t)
      # record
      self.load_value_episode.append(self.load_value_current)

      # .............. verify if solutions are the same with the given conditions ..............
      for i in self.state_line_status.keys():
        if abs(self.state_line_status[i] - sol_line_status[i][0]) > 1e-6:
          print('GYM_Power: solutions and conditions are not the same for ' + i)

      #.............. update system status and observation .............
      for i in sol_load_status.keys():
        self.state_load_status[i] = sol_load_status[i][0]

      # ............. initialize observations................
      self.current_observ = []
      for i in self.state_line_status.keys():
        self.current_observ.append(self.state_line_status[i])

    # ------------------------- if infeasible TODO: maybe use try structure -------------------
    else:
      print("GYM_Power: Current generated disturbance makes the system infeasible")
      print("GYM_Power: Disturbance is: ", self.disturbance)
      sys.exit()

    # convert load status to a list
    _load_status = self.dict2list(self.state_load_status)

    return np.array(self.current_observ), np.array(_load_status)


  def step(self, action):
    """
    Apply the given actions to the environment for one step
    """
    # check is action format is correct
    assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    # ===================== parse action from GYM format to optimization format =====================
    # get current line status; remember to use copy so that modification of "action_line" will not impact "self.state_line_status"
    action_line = self.state_line_status.copy()
    if action == 1:
      action_line['line_118'] = int(not (action_line['line_118']))
    elif action == 2:
      action_line['line_119'] = int(not (action_line['line_119']))
    elif action == 3:
      action_line['line_120'] = int(not (action_line['line_120']))
    elif action == 4:
      action_line['line_121'] = int(not (action_line['line_121']))
    elif action == 5:
      action_line['line_122'] = int(not (action_line['line_122']))
    elif action == 6:
      action_line['line_123'] = int(not (action_line['line_123']))
    elif action == 7:
      action_line['line_124'] = int(not (action_line['line_124']))
    elif action == 8:
      action_line['line_125'] = int(not (action_line['line_125']))
    elif action == 9:
      action_line['line_126'] = int(not (action_line['line_126']))
    elif action == 10:
      action_line['line_127'] = int(not (action_line['line_127']))
    elif action == 11:
      action_line['line_128'] = int(not (action_line['line_128']))
    elif action == 12:
      action_line['line_129'] = int(not (action_line['line_129']))
    elif action == 13:
      action_line['line_130'] = int(not (action_line['line_130']))
    elif action == 14:
      action_line['line_131'] = int(not (action_line['line_131']))
    elif action == 15:
      action_line['line_132'] = int(not (action_line['line_132']))
    elif action == 0:
      pass
    else:
      print("GYM_Power: Out of action space")


    # ===================== determine termination condition and rewards =====================
    # first check if this is the last step in this episode
    if self.exploration_index == self.exploration_total:
      done = True  # we are done with this episode
      reward = 0
    else:
      done = False

      # =====================  solve for load status =====================
      self.sim_case.solve_load_pickup(action_line, self.state_load_status)
      opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')  #
      # opt = pm.SolverFactory("cplex")
      opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
      results = opt.solve(self.sim_case.model, tee=False)

      # ----------- if infeasible, do nothing -------------
#       if results['solver'][0]['Termination condition'].key == 'infeasible':
      if results.solver.termination_condition == TerminationCondition.infeasible:
        reward = -1000
        self.solver_condition = 'infeasible'
        # if infeasible, we do not update the system status
        # add the current load status into the episode list
        self.load_value_episode.append(self.load_value_current)

      # --------------- if optimal, update status, observations and determine rewards ----------------
#       elif results['solver'][0]['Termination condition'].key == 'optimal':
      elif results.solver.termination_condition == TerminationCondition.optimal:
        self.solver_condition = 'optimal'
        # if optimal, get results
        sol_load_status = self.sim_case.get_solution_2d('rho', self.sim_case.iter_bus, self.sim_case.iter_time)
        sol_line_status = self.sim_case.get_solution_2d('ul', self.sim_case.iter_line, self.sim_case.iter_time)

        # ...............verify if solutions are the same with the given conditions............
        for i in action_line.keys():
          if abs(action_line[i] - sol_line_status[i][0]) > 1e-6:
            print('GYM_Power: solutions and conditions are not the same for ' + i)

        # ................. update system status and observation .................
        # retrieve bus-indexed status
        for i in sol_load_status.keys():
          self.state_load_status[i] = sol_load_status[i][0]
        # retrieve line-indexed status
        for i in sol_line_status.keys():
          self.state_line_status[i] = sol_line_status[i][0]

          # fresh observations
          self.current_observ = []
          for i in self.state_line_status.keys():
            self.current_observ.append(self.state_line_status[i])

        # ................. get new total load value and determine rewards .................
        _temp_load_t = []
        for t in self.sim_case.iter_time:
          _temp_load = 0
          for i in self.sim_case.iter_bus:
            id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
            idx = np.where(id == self.ppc['bus'][:, 0])[0][0]
            _temp_load = _temp_load + sol_load_status[i][t] * self.ppc['bus'][idx, 4]
          _temp_load_t.append(_temp_load)
        load_value_new = sum(_temp_load_t)
        # compare the current and new load value to determine the rewards
        if load_value_new > self.load_value_current:
          reward = 150
        elif load_value_new == self.load_value_current:
          reward = -10
        elif load_value_new < self.load_value_current:
          reward = -100
        else:
          print('GYM_Power: impossible load condition')
          sys.exit()
        # update load value and append it into the episode list
        self.load_value_current = load_value_new
        self.load_value_episode.append(self.load_value_current)

      else:
        print('GYM_Power: Unknown solver condition')
        sys.exit()

      # update index
      self.exploration_index += 1

    # convert load status to a list
    _load_status = self.dict2list(self.state_load_status)

    return np.array(self.current_observ), reward, done, np.array(_load_status)


  def dict2list(self, dict):
    """
    convert an order dict to a list
    """
    a = []
    for k in dict.keys():
      a.append(dict[k])
    return a


  def render(self, mode='human'):
    pass


  def close(self):
    pass








class RestorationDisVarConEnv(gym.Env):
  """
  Restoration env with tieline and reactive power dispatch control
  """
  metadata = {'render.modes': ['human']}

  def __init__(self, max_disturbance, min_disturbance):
    # ================== initialize network data and status ====================
    # network data
    self.ppc = case33_tieline_DG()
    num_line = len(self.ppc['iter_line'])
    num_bus = len(self.ppc['iter_bus'])
    num_tieline = len(self.ppc['tieline'])
    num_varcon = len(self.ppc['varcon'])
    varcon_lower_limit = []
    varcon_upper_limit = []
    for i in range(1, len(self.ppc['iter_gen'])):  # minus the substation
      varcon_lower_limit.append(self.ppc['gen'][i, 4])
      varcon_upper_limit.append(self.ppc['gen'][i, 5])
    P_lower_limit = []
    P_upper_limit = []
    Q_lower_limit = []
    Q_upper_limit = []
    for i in range(num_varcon):  # minus the substation
      P_lower_limit.append(-self.ppc['line'][i, 4])
      P_upper_limit.append(self.ppc['gen'][i, 4])
      Q_lower_limit.append(-self.ppc['line'][i, 5])
      Q_upper_limit.append(self.ppc['gen'][i, 5])

    # exploration number is equal to tie-line numbers
    # consider that
    # (1) range(5) = [0, 1, 2, 3, 4, 5];
    # (2) in optimization we model initial step and enforce all tielines to be zero
    # we set exploration_total to be five
    self.exploration_total = num_tieline
    self.exploration_seq_idx = [i for i in range(self.exploration_total + 1)]

    # action number is equal to the tie-line plus one (do nothing option)
    self.action_space = spaces.Dict({
      'tieline': spaces.Discrete(num_tieline + 1),
      'varcon': spaces.Box(np.array(varcon_lower_limit), np.array(varcon_upper_limit), dtype=np.float32)
    })
    self.observation_space = spaces.Box(np.array([0] * (num_line)), np.array([1] * (num_line)), dtype=np.int)

    # =========== disturbance upper bound =============
    self.max_disturbance = max_disturbance
    self.min_disturbance = min_disturbance
    print("max_disturbance is {}".format(max_disturbance))
    print("min_disturbance is {}".format(min_disturbance))

    # ============ voltage specifications ============
    self.VS = 1.05
    self.dV = 0.05

    self.seed()
    self.viewer = None
    self.state = None

    self.steps_beyond_done = None


  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]


  def reset(self, disturbance=None):
    """
    Here we should setup the episode control for the total time horizon, which will be independent of individual segment simulation
    """
    # ================= define system states for a new episode ===============
    # ordered dict is very important since learning data is in the form of matrices
    self.state_line_status = OrderedDict()
    self.state_load_status = OrderedDict()
    self.state_flow_p = OrderedDict()
    self.state_flow_q = OrderedDict()
    self.state_varcon = OrderedDict()

    # initialize a list to store the load status during an episode
    self.load_value_current = 0
    self.load_value_episode = []

    # index to determine the instants
    self.exploration_index = 0
    # print('Exploration index is reset to one.')

    # ================== generate disturbance ====================
    # generate random disturbance if no specific one is given
    if disturbance == None:
      disturbance_set = ['line_3', 'line_4', "line_5", 'line_6', 'line_7', 'line_8', 'line_9', 'line_10', 'line_11', 'line_12', 'line_13', 'line_14',
                         'line_15', 'line_16', 'line_17', 'line_18', 'line_19', 'line_20', 'line_21', 'line_22', 'line_23', 'line_24', 'line_25',
                         'line_26', 'line_27', 'line_28', 'line_29', 'line_30', 'line_31', 'line_32']
      # generate disturbance upper bound for this episoid
      num_disturbance = random.randint(self.min_disturbance, self.max_disturbance)
      # record generated disturbance
      self.disturbance = []
      for i in range(num_disturbance):
        # generate one line outage at a time
        random_disturbance = random.choice(disturbance_set)
        # record
        self.disturbance.append(random_disturbance)
        # remove from the set
        disturbance_set.remove(random_disturbance)
    else:
      self.disturbance = disturbance

    # =============== initialize the line and var control for optimization ===============
    # initialize line status
    for i in self.ppc['iter_line']:
      self.state_line_status[i] = 1
    for i in self.ppc['tieline']:
      self.state_line_status[i] = 0   # initially controls are zero
    # update line status
    for i in self.disturbance:
      self.state_line_status[i] = 0

    # initialize var control as zero
    for i in self.ppc['varcon']:
      self.state_varcon[i] = 0  # initially controls are zero

    # =============== solve the load status given line status  ===============
    self.sim_case = OutageManage()
    self.sim_case.data_preparation(self.ppc, self.disturbance, VS=self.VS, dV=self.dV)
    self.sim_case.solve_load_pickup_varcon(self.state_line_status, self.state_varcon)
    opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
    opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
    results = opt.solve(self.sim_case.model, tee=False)

    # check feasibility
    # ------------------------- if optimal, update system status -------------------
    # if results['solver'][0]['Termination condition'].key == 'optimal':
    if results.solver.termination_condition == TerminationCondition.optimal:
      # ................get optimization results................
      sol_load_status = self.sim_case.get_solution_2d('rho', self.sim_case.iter_bus, self.sim_case.iter_time)
      sol_line_status = self.sim_case.get_solution_2d('ul', self.sim_case.iter_line, self.sim_case.iter_time)
      sol_gen_q = self.sim_case.get_solution_2d('q', self.sim_case.iter_gen, self.sim_case.iter_time)
      sol_flow_p = self.sim_case.get_solution_2d('P', self.sim_case.iter_line, self.sim_case.iter_time)
      sol_flow_q = self.sim_case.get_solution_2d('Q', self.sim_case.iter_line, self.sim_case.iter_time)
      # .................calculate current total load............
      _temp_load_t = []
      for t in self.sim_case.iter_time:
        _temp_load = []
        for i in self.sim_case.iter_bus:
          id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
          idx = np.where(id == self.ppc['bus'][:, 0])[0][0]
          _temp_load.append(sol_load_status[i][t] * self.ppc['bus'][idx, 4])
        _temp_load_t.append(sum(_temp_load))
      # update the current load value
      self.load_value_current = sum(_temp_load_t)
      # record
      self.load_value_episode.append(self.load_value_current)

      # .............. verify if solutions are the same with the given conditions ..............
      for i in self.state_line_status.keys():
        if abs(self.state_line_status[i] - sol_line_status[i][0]) > 1e-6:
          print('GYM_Power: solutions and conditions are not the same for ' + i)
      for i in self.state_varcon.keys():
        if abs(self.state_varcon[i] - sol_gen_q[i][0]) > 1e-4:
          print('GYM_Power: solutions and conditions are not the same for ' + i)

      #.............. update system status and observation .............
      # retrieve bus-indexed status: load and voltage
      for i in sol_load_status.keys():
        self.state_load_status[i] = sol_load_status[i][0]
      # retrieve line-indexed status: power flow values
      for i in sol_flow_p.keys():
        self.state_flow_p[i] = sol_flow_p[i][0]
        self.state_flow_q[i] = sol_flow_q[i][0]

      # ............. initialize observations................
      self.current_observ = {'line': np.array([]), 'load': np.array([]), 'flow_p': np.array([]), 'flow_q': np.array([])}
      for i in self.state_line_status.keys():
        self.current_observ['line'] = np.append(self.current_observ['line'], round(self.state_line_status[i]))
      for i in self.state_load_status.keys():
        self.current_observ['load'] = np.append(self.current_observ['load'], round(self.state_load_status[i]))
      for i in self.state_flow_p.keys():
        self.current_observ['flow_p'] = np.append(self.current_observ['flow_p'], self.state_flow_p[i])
      for i in self.state_flow_q.keys():
        self.current_observ['flow_q'] = np.append(self.current_observ['flow_q'], self.state_flow_q[i])

    # ------------------------- if infeasible TODO: maybe use try structure -------------------
    else:
      print("GYM_Power: Current generated disturbance makes the system infeasible")
      print("GYM_Power: Disturbance is: ", self.disturbance)
      sys.exit()

    return self.current_observ


  def step(self, action, logger=None):
    """
    Apply the given actions to the environment for one step
    Action here is a dictionary with "tieline" and "varcon"
    """
    # check is action format is correct
    # assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

    # ===================== parse action from GYM format to optimization format =====================
    # get current line status; remember to use copy so that modification of "action_line" will not impact "self.state_line_status"
    action_line = self.state_line_status.copy()
    if action['tieline'] == 1:
      action_line['line_33'] = int(not (action_line['line_33']))
    elif action['tieline'] == 2:
      action_line['line_34'] = int(not (action_line['line_34']))
    elif action['tieline'] == 3:
      action_line['line_35'] = int(not (action_line['line_35']))
    elif action['tieline'] == 4:
      action_line['line_36'] = int(not (action_line['line_36']))
    elif action['tieline'] == 5:
      action_line['line_37'] = int(not (action_line['line_37']))
    elif action['tieline'] == 0:
      pass
    else:
      print("GYM_Power: Out of action space")

    action_varcon = {}
    idx = 0
    for i in self.ppc['varcon']:
      action_varcon[i] = action['varcon'][idx]
      idx = idx + 1

    # ===================== determine termination condition and rewards =====================
    # first check if this is the last step in this episode
    if self.exploration_index == self.exploration_total:
      done = True  # we are done with this episode
      reward = 0
    else:
      done = False

      # =====================  solve for load status =====================
      self.sim_case.solve_load_pickup_varcon(action_line, action_varcon, self.state_load_status)
      opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
      opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
      results = opt.solve(self.sim_case.model, tee=False)

      # ----------- if infeasible, do nothing -------------
      # if results['solver'][0]['Termination condition'].key == 'infeasible':
      if results.solver.termination_condition == TerminationCondition.infeasible:
        reward = -1000
        self.solver_condition = 'infeasible'
        # if infeasible, we do not update the system status
        # add the current load status into the episode list
        self.load_value_episode.append(self.load_value_current)

      # --------------- if optimal, update status, observations and determine rewards ----------------
      # elif results['solver'][0]['Termination condition'].key == 'optimal':
      elif results.solver.termination_condition == TerminationCondition.optimal:
        self.solver_condition = 'optimal'
        # if optimal, get results
        sol_load_status = self.sim_case.get_solution_2d('rho', self.sim_case.iter_bus, self.sim_case.iter_time)
        sol_line_status = self.sim_case.get_solution_2d('ul', self.sim_case.iter_line, self.sim_case.iter_time)
        sol_gen_q = self.sim_case.get_solution_2d('q', self.sim_case.iter_gen, self.sim_case.iter_time)
        sol_flow_p = self.sim_case.get_solution_2d('P', self.sim_case.iter_line, self.sim_case.iter_time)
        sol_flow_q = self.sim_case.get_solution_2d('Q', self.sim_case.iter_line, self.sim_case.iter_time)

        # ...............verify if solutions are the same with the given conditions............
        for i in action_line.keys():
          if abs(action_line[i] - sol_line_status[i][0]) > 1e-6:
            print('GYM_Power: solutions and conditions of line status are not the same for ' + i)
            if logger is not None:
              logger.info('GYM_Power: solutions and conditions for line status are not the same for ' + i)
        for i in action_varcon.keys():
          if abs(action_varcon[i] - sol_gen_q[i][0]) > 1e-4:
            print('GYM_Power: solutions and conditions of var are not the same for ' + i)
            if logger is not None:
              logger.info('GYM_Power: solutions and conditions of var are not the same for ' + i)
              logger.info('action: {}; solution: {}'.format(action_varcon[i], sol_gen_q[i][0]))
        # ................. update system status and observation .................
        # retrieve bus-indexed status
        for i in sol_load_status.keys():
          self.state_load_status[i] = sol_load_status[i][0]
        # retrieve line-indexed status
        for i in sol_line_status.keys():
          self.state_line_status[i] = sol_line_status[i][0]
          self.state_flow_p[i] = sol_flow_p[i][0]
          self.state_flow_q[i] = sol_flow_q[i][0]

          # ............. fresh observations................
          self.current_observ = {'line': np.array([]), 'load': np.array([]), 'flow_p': np.array([]), 'flow_q': np.array([])}
          for i in self.state_line_status.keys():
            self.current_observ['line'] = np.append(self.current_observ['line'], round(self.state_line_status[i]))
          for i in self.state_load_status.keys():
            self.current_observ['load'] = np.append(self.current_observ['load'], round(self.state_load_status[i]))
          for i in self.state_flow_p.keys():
            self.current_observ['flow_p'] = np.append(self.current_observ['flow_p'], self.state_flow_p[i])
          for i in self.state_flow_q.keys():
            self.current_observ['flow_q'] = np.append(self.current_observ['flow_q'], self.state_flow_q[i])

        # ................. get new total load value and determine rewards .................
        _temp_load_t = []
        for t in self.sim_case.iter_time:
          _temp_load = 0
          for i in self.sim_case.iter_bus:
            id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
            idx = np.where(id == self.ppc['bus'])[0][0]
            _temp_load = _temp_load + sol_load_status[i][t] * self.ppc['bus'][idx, 4]
          _temp_load_t.append(_temp_load)
        load_value_new = sum(_temp_load_t)
        # compare the current and new load value to determine the rewards
        if load_value_new > self.load_value_current:
          reward = 150
        elif load_value_new == self.load_value_current:
          reward = -10
        elif load_value_new < self.load_value_current:
          reward = -100
        else:
          print('GYM_Power: impossible load condition')
          sys.exit()
        # update load value and append it into the episode list
        self.load_value_current = load_value_new
        self.load_value_episode.append(self.load_value_current)

      else:
        print('GYM_Power: Unknown solver condition')
        sys.exit()

      # update index
      self.exploration_index += 1

    return self.current_observ, reward, done, {}


  def view_grid(self):
    G = nx.Graph()
    for i in self.ppc['line_bus']:
      if self.state_line_status[i] == 0:
        continue
      elif self.state_line_status[i] == 1:
        _bus_from = self.ppc['line_bus'][i][0]
        _bus_to = self.ppc['line_bus'][i][1]
        G.add_edge(_bus_from, _bus_to)
      else:
        print("Not a possible scenario")
    plt.rcParams.update({'font.family': 'Arial'})
    plt.figure(figsize=(15, 8))
    # ax = plt.gca()
    # ax.set_title('Step {}'.format(self.exploration_index))
    if_connected = nx.is_connected(G)
    nx.draw(G, self.ppc['pos'], with_labels=True, font_weight='bold')


  def render(self, mode='human'):
    pass


  def close(self):
    pass
