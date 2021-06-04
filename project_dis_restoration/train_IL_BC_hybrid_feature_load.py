import time
from datetime import datetime
import os
import sys
import logging
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from NN_model import DNN_TieLine, DNN_VarCon

import gym
from gym import spaces
import gym_power_res  # This import command is very important for gym to recognize the package "gym-power-res"
import pyomo.environ as pm
from pyomo.opt import SolverStatus, TerminationCondition
from gym_power_res.envs.DS_pyomo import OutageManage, SolutionDict
from gym_power_res.sorces import ScoreLogger, SuccessLogger
from gym_power_res.envs.data_test_case import case33_tieline, case33_tieline_DG


# Parameters
ENV_NAME_1 = "RestorationDisEnv-v0"
ENV_NAME_2 = "RestorationDisVarConEnv-v0"
NUM_TRAIN_EPISODES = 500
NUM_TOTAL_EPISODES = 500
MAX_DISTURBANCE = 2
MIN_DISTURBANCE = 2

class DictList(dict):
    def __setitem__(self, key, value):
        try:
            # Assumes there is a list on the key
            self[key].append(value)
        except KeyError: # If it fails, because there is no key
            super(DictList, self).__setitem__(key, value)
        except AttributeError: # If it fails because it is not a list
            super(DictList, self).__setitem__(key, [self[key], value])


def logger_obj(logger_name, level=logging.DEBUG, verbose=0):
    """
    Method to return a custom logger with the given name and level
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    format_string = ("%(asctime)s - %(levelname)s - %(funcName)s (%(lineno)d):  %(message)s")
    datefmt = '%Y-%m-%d %I:%M:%S %p'
    log_format = logging.Formatter(format_string, datefmt)

    # Creating and adding the file handler
    file_handler = logging.FileHandler(logger_name, mode='a')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if verbose == 1:
        # Creating and adding the console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)

    return logger


# here we combine NN model and agent models
class Agent:
    def __init__(self, env, output_path, log_level=logging.DEBUG):
        # retrieve necessary parameters from network data
        num_line = len(env.ppc['iter_line'])
        num_bus = len(env.ppc['iter_bus'])
        num_tieline = len(env.ppc['tieline'])
        num_varcon = len(env.ppc['varcon'])

        self.nn_tieline = DNN_TieLine(num_line, num_tieline + 1)
        # self.nn_varcon = [DNN_VarCon(num_line * 2, num_varcon) for index in range(num_tieline + 1)]
        self.nn_varcon = [DNN_VarCon(num_bus, num_varcon) for index in range(num_tieline + 1)]

        self.total_episode = 0

        self.warmstart = True

        self.if_logging = True

        self.if_timer = True

        # create dir for results and logs
        now = datetime.now()
        dt_string = now.strftime("__%Y_%m_%d_%H_%M")
        self.dt_string = dt_string
        # check if the dir is given
        if output_path is not None:
            # if given, check if the saving directory exists
            # if not given, create dir
            if not os.path.isdir(output_path):
                os.makedirs(output_path)
            log_output_path_name = output_path + '/' + "log" + dt_string + '.log'
        elif output_path is None:
            # if dir is not given, save results at root dir
            output_path = os.getcwd()
            log_output_path_name = output_path + '/' + "log" + dt_string + '.log'
        self.logger = logger_obj(logger_name=log_output_path_name, level=log_level)  # log for debuging
        self.success_logger = SuccessLogger(ENV_NAME_2, output_path, title='Hybrid Behavior Cloning')  # log for performance evaluation


    def get_expert_policy(self, env, s0):
        """
        get the expert policy by solving the full-period restoration problem
        """
        if self.if_timer == True:
            print("Begin quering expert")
            stime = time.time()

        expert = OutageManage()
        expert.data_preparation(env.ppc, env.disturbance)
        expert.solve_network_restoration_varcon(len(env.exploration_seq_idx), env.state_line_status, env.state_varcon, env.state_load_status)
        opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
        opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
        expert_result = opt.solve(expert.model, tee=False)

        # if optimal, retrieve the expert actions
        # if expert_result['solver'][0]['Termination condition'].key == 'optimal':
        if expert_result.solver.termination_condition == TerminationCondition.optimal:
            # status to retrieve high-level policy
            expert_sol_load_status = expert.get_solution_2d('rho', expert.iter_bus, expert.iter_time)
            expert_sol_line_status = expert.get_solution_2d('ul', expert.iter_line, expert.iter_time)
            # get load profile
            expert_load_profile_bin, expert_load_profile_P = self.load_profile(env, expert_sol_load_status)
            # status to retrieve low-level policy
            # low-level action
            expert_sol_gen_q = expert.get_solution_2d('q', expert.iter_gen, expert.iter_time)
            # low-level state
            expert_sol_flow_p = expert.get_solution_2d('P', expert.iter_line, expert.iter_time)
            expert_sol_flow_q = expert.get_solution_2d('Q', expert.iter_line, expert.iter_time)
        else:
            # TODO: what to do when the expert problem is infeasible
            self.logger.info("expert does not return optimal solution under disturbance {}".format(env.disturbance))
            print("expert is not able to provide action under disturbance {}".format(env.disturbance))

        if self.if_timer == True:
            print("Complete quering expert using {}".format(time.time() - stime))

        # =================== generate trajectory ======================
        # s0 -> a1 -> s1 -> a2 -> s2 -> a3 -> s3 -> a4 -> s4 -> a5 -> s5
        # in this case, we need to store (s0, a1), (s1, a2), (s2, a3), (s3, a4), (s4, a5)
        # To create low-level expert dict, we use a list to build an ordered data to reflect the trajectory.
        expert_tieline_trajectory = []
        expert_varcon_trajectory = []

        # ------------ get state of initial episode -----------
        # high-level state
        s0_line = s0['line']
        s0_load = s0['load']
        # s0_pq = np.concatenate((s0['flow_p'], s0['flow_q']), axis=0)

        # -------------- explore the episode sequence from 1 to the total exploration (note that 0 denotes the initial condition) ----------
        for step in env.exploration_seq_idx[-env.exploration_total:]:
            # s0_pq = tuple(s0_pq)
            s0_line = tuple(s0_line)
            s0_load = tuple(s0_load)

            # get tieline status (note that tieline status and actions are different)
            tieline_status_0 = s0_line[-5:]
            # get tieline action w.r.t. s0 by minus tieline status
            tieline_status_1 = []
            for i in env.ppc["tieline"]:
                tieline_status_1.append(round(expert_sol_line_status[i][step]))
            # minus the list to get the tieline action
            a1_tieline = self.convert_tieline_action(tieline_status_1, tieline_status_0)
            # # record to logger
            if self.if_logging == True:
                self.logger.info("expert high-level action at step {} is {}".format(step, a1_tieline))

            # get var control action
            a1_varcon = np.array([])
            for i in env.ppc["varcon"]:
                a1_varcon = np.append(a1_varcon, expert_sol_gen_q[i][step])
            if self.if_logging == True:
                self.logger.info("expert low-level action at step {} is {}".format(step, a1_varcon))

            # record trajectory
            expert_tieline_trajectory.append((s0_line, a1_tieline))
            # expert_varcon_trajectory.append((s0_pq, a1_varcon))
            expert_varcon_trajectory.append((s0_load, a1_varcon))

            # update state under action a1
            s0_line = []
            for i in expert_sol_line_status.keys():
                s0_line.append(round(expert_sol_line_status[i][step]))
            s0_load = []
            for i in expert_sol_load_status.keys():
                s0_load.append(round(expert_sol_load_status[i][step]))
            # state for var control
            # s0_pq = []
            # for k in expert_sol_flow_p.keys():
            #     s0_pq.append(expert_sol_flow_p[k][step])
            # for k in expert_sol_flow_q.keys():
            #     s0_pq.append(expert_sol_flow_q[k][step])

        # ================== parse trajectory into high-level policy ===================
        # since we may have the same state, we will store expert action in new DictList object
        self.expert_high = DictList()
        for step in env.exploration_seq_idx[-env.exploration_total:]:
            s0_line = expert_tieline_trajectory[step - 1][0]
            a1_tieline = expert_tieline_trajectory[step - 1][1]
            if s0_line not in self.expert_high:
                self.expert_high[s0_line] = []
            self.expert_high[s0_line].append(a1_tieline)

        # ================== parse trajectory into low-level policy ====================
        self.expert_low = {}
        # --------- explore the episode sequence to get low-level state(0)-action(1) pairs, where the number denotes the step index--------
        for step in env.exploration_seq_idx[-env.exploration_total:]:
            # get high-level action as the key for low-level policy
            a1_tieline = expert_tieline_trajectory[step - 1][1]
            if a1_tieline not in self.expert_low.keys():
                self.expert_low[a1_tieline] = DictList()
                self.logger.info("Record high-level action: {}".format(a1_tieline))
            # store low-level (state0, action1) pairs
            s0_load = expert_varcon_trajectory[step - 1][0]
            s0_load = tuple(s0_load)
            a1_varcon = expert_varcon_trajectory[step - 1][1]
            if s0_load not in self.expert_low[a1_tieline].keys():
                self.expert_low[a1_tieline][s0_load] = []
                self.logger.info("Record low-level action: {}".format(s0_load))
            self.expert_low[a1_tieline][s0_load].append(a1_varcon)


    def load_profile(self, env, load_status):
        load_profile_bin = []
        load_profile_P = []
        total_step = len(load_status['bus_1'])
        for t in range(total_step):
            _tem_bin = []
            _tem_P = []
            for i in load_status:
                id = int(i[i.find('_') + 1:]) - 1  # get the matrix index from the component name
                _tem_bin.append(load_status[i][t])
                _tem_P.append(load_status[i][t] * env.ppc['bus'][id, 4])
            load_profile_bin.append(sum(_tem_bin))
            load_profile_P.append(sum(_tem_P))
        return load_profile_bin, load_profile_P


    def convert_tieline_action(self, s1_tieline, s0_tieline):
        """
        :param s1_tieline: current tieline status
        :param s0_tieline: previsou tieline status
        :return: integer action from 0 to 5
        """
        d = []  # tieline change
        for k in range(len(s1_tieline)):
            d.append(abs(s1_tieline[k] - s0_tieline[k]))

        # since we only allow one action at a time, there are only several scenarios
        # We traverse all scenarios
        if d == [0, 0, 0, 0, 0]:
            a = 0  # no action
        elif d == [1, 0, 0, 0, 0]:
            a = 1
        elif d == [0, 1, 0, 0, 0]:
            a = 2
        elif d == [0, 0, 1, 0, 0]:
            a = 3
        elif d == [0, 0, 0, 1, 0]:
            a = 4
        elif d == [0, 0, 0, 0, 1]:
            a = 5
        else:
            print("Multiple actions exist in one step!")
            sys.exit()
        return a


    def policy_hybrid(self, env, s_high, s_low, mode):
        # define the action dict based on GYM format
        action = {'tieline': 0, 'varcon': np.array([0, 0, 0, 0, 0, 0])}

        # if mode is training and warm start is true, we use expert action
        if mode == "train" and self.warmstart == True:
            # first we get high-level action
            ah = self.expert_high[s_high][0]  # in case multiple actions have the same state
            action['tieline'] = ah
            # then we get low-level action
            # key_low = self.find_float_key(self.expert_low[ah], s_low)
            if self.if_logging == True:
                self.logger.info("Looking for low-level state: {}".format(s_low))
            action['varcon'] = self.expert_low[ah][s_low][0]
            if self.if_logging == True:
                self.logger.info("Using expert high-level action: {}".format(ah))
                self.logger.info("Using expert low-level action: {}".format(action['varcon']))

        # either mode is test or warm start is false, we use approximator
        elif mode == "test" or self.warmstart == False:
            # get high-level approximated policy
            ah_prob = self.nn_tieline.model.predict(np.reshape(s_high, (-1, self.nn_tieline.input_shape)))
            ah = ah_prob.argmax()
            action['tieline'] = ah
            # then we get low-level action
            al = self.nn_varcon[ah].model.predict(np.reshape(s_low, (-1, self.nn_varcon[ah].input_shape)))
            al = np.reshape(al, -1)
            # apply bounds into the predicted actions
            al = np.minimum(env.action_space['varcon'].high, al)
            al = np.maximum(env.action_space['varcon'].low, al)
            action['varcon'] = al
            if self.if_logging == True:
                self.logger.info("Using predicted high-level action: {}".format(ah))
                self.logger.info("Using predicted low-level action: {}".format(al))

        else:
            print("Policy choice logics out of the space")

        return action



    def find_float_key(self, dict, near_key):
        knear = np.array(near_key)
        key_list = list(dict.keys())
        distant_list = np.array([])
        for k in key_list:
            k = np.array(k)
            distant_list = np.append(distant_list, np.linalg.norm(knear - k))
        return key_list[distant_list.argmin()]


    def run_train(self, env, s0):

        if self.if_timer == True:
            print("Begin exploration")
            stime = time.time()

        # ============== retrieve the initial states from the environment ==================
        # ------------ get state of initial episode -----------
        # high-level state
        s0_line = s0['line']
        s0_load = s0['load']
        # s0_pq = np.concatenate((s0['flow_p'], s0['flow_q']), axis=0)

        # ============== apply expert/stochastic policy into the env and collect samples for training ==================
        for step in env.exploration_seq_idx[-env.exploration_total:]:

            if self.if_logging == True:
                self.logger.info("-----Step {}-----".format(step))

            s0_line = tuple(s0_line)
            s0_load = tuple(s0_load)
            a_gym = self.policy_hybrid(env, s0_line, s0_load, mode='train')

            # random action also works, but slower since you can bump into walls
            # a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            s_gym, _, _, _ = env.step(a_gym)

            # parse gym-formatted action
            a1_tieline = a_gym['tieline']
            a1_varcon = a_gym['varcon']
            # collect (state,action) pair
            self.nn_tieline.collect(s0_line, a1_tieline)
            self.nn_varcon[a1_tieline].collect(s0_load, a1_varcon)
            if self.if_logging == True:
                self.logger.info("collected high-level s (last five) {}".format(s0_line[-5:]))
                self.logger.info("collected high-level a {}".format(a1_tieline))
                self.logger.info("collected low-level a {}".format(a1_varcon))

            # update state
            s0_line = s_gym['line']
            s0_load = s_gym['load']
            # s0_pq = np.concatenate((s_gym['flow_p'], s_gym['flow_q']), axis=0)

        if self.if_timer == True:
            print("Complete exploration using {}".format(time.time() - stime))

        if self.if_timer == True:
            print("Begin training")
            stime = time.time()

        if self.total_episode >= 1:
            loss = self.nn_tieline.end_collect()
            for nn_k in self.nn_varcon:
                nn_k.end_collect()

        if self.if_timer == True:
            print("Complete training using {}".format(time.time() - stime))


    def run_test(self, env, s0):

        if self.if_timer == True:
            print("Begin solving benchmark")
            stime = time.time()

        # ================ calculate Benchmark value to normalize the restoration as ratio ==================
        self.logger.info("-------------------Run_test begin--------------------")
        self.logger.info("The testing disturbance is {}".format(env.disturbance))

        # if_logging is true, we also compare benchmark with tieline-only case
        if self.if_logging == True:
            ppc_1 = case33_tieline()
            expert_1 = OutageManage()
            expert_1.data_preparation(ppc_1, env.disturbance, 1.05, 0.05)
            expert_1.solve_network_restoration(len(env.exploration_seq_idx), env.state_line_status, env.state_load_status)
            opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
            opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
            expert_result_1 = opt.solve(expert_1.model, tee=False)
            # if optimal, get the optimal value
            if expert_result_1.solver.termination_condition == TerminationCondition.optimal:
                _expert_sol_load_status = expert_1.get_solution_2d('rho', expert_1.iter_bus, expert_1.iter_time)
                _expert_load_profile_bin, _expert_load_profile_P = self.load_profile(env, _expert_sol_load_status)
                self.logger.info("Tieline-only Benchmark return optimal solution: {}".format(_expert_load_profile_bin))
                self.logger.info("Tieline-only Benchmark return optimal solution: {}".format(_expert_load_profile_P))
                goal_1 = sum(_expert_load_profile_P) - _expert_load_profile_P[0] * len(expert_1.iter_time)
                self.logger.info("Tieline-only Benchmark goal: {}".format(goal_1))
            else:
                self.logger.info("This scenario is infeasible; continue to the next training loop")
                Flag_continue = True

        ppc = case33_tieline_DG()
        expert = OutageManage()
        expert.data_preparation(ppc, env.disturbance, 1.05, 0.05)
        expert.solve_network_restoration_varcon(len(env.exploration_seq_idx), env.state_line_status, env.state_varcon, env.state_load_status)
        opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
        opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
        expert_result = opt.solve(expert.model, tee=False)
        # if optimal, get the optimal value
        if expert_result.solver.termination_condition == TerminationCondition.optimal:
            expert_sol_load_status = expert.get_solution_2d('rho', expert.iter_bus, expert.iter_time)
            expert_sol_line_status = expert.get_solution_2d('ul', expert.iter_line, expert.iter_time)
            expert_sol_gen_q = expert.get_solution_2d('q', expert.iter_gen, expert.iter_time)
            expert_load_profile_bin, expert_load_profile_P = self.load_profile(env, expert_sol_load_status)
            if self.if_logging == True:
                self.logger.info("Benchmark return optimal solution: {}".format(expert_load_profile_bin))
                self.logger.info("Benchmark return optimal solution: {}".format(expert_load_profile_P))
                self.logger.info("Benchmark return optimal line 33: {}".format(expert_sol_line_status['line_33']))
                self.logger.info("Benchmark return optimal line 34: {}".format(expert_sol_line_status['line_34']))
                self.logger.info("Benchmark return optimal line 35: {}".format(expert_sol_line_status['line_35']))
                self.logger.info("Benchmark return optimal line 36: {}".format(expert_sol_line_status['line_36']))
                self.logger.info("Benchmark return optimal line 37: {}".format(expert_sol_line_status['line_37']))
                self.logger.info("Benchmark return optimal gen 1: {}".format(expert_sol_gen_q['gen_1']))
                self.logger.info("Benchmark return optimal gen 2: {}".format(expert_sol_gen_q['gen_2']))
                self.logger.info("Benchmark return optimal gen 3: {}".format(expert_sol_gen_q['gen_3']))
                self.logger.info("Benchmark return optimal gen 4: {}".format(expert_sol_gen_q['gen_4']))
                self.logger.info("Benchmark return optimal gen 5: {}".format(expert_sol_gen_q['gen_5']))
                self.logger.info("Benchmark return optimal gen 6: {}".format(expert_sol_gen_q['gen_6']))
                self.logger.info("Benchmark return optimal gen 7: {}".format(expert_sol_gen_q['gen_7']))
            goal = sum(expert_load_profile_P) - expert_load_profile_P[0] * len(expert.iter_time)  # we need to subtract the base load
            self.logger.info("Benchmark goal: {}".format(goal))
        else:
            self.logger.info("This scenario is infeasible; continue to the next training loop")
            Flag_continue = True

        if self.if_timer == True:
            print("Complete solving benchmark using {}".format(time.time() - stime))

        if self.if_timer == True:
            print("Begin testing")
            stime = time.time()

        # ============== retrieve the initial states from the environment ==================
        # ------------ get state of initial episode -----------
        # high-level state
        s0_line = s0['line']
        s0_load = s0['load']
        # s0_pq = np.concatenate((s0['flow_p'], s0['flow_q']), axis=0)

        # ============== run agent using trained policy approximator ==================
        for step in env.exploration_seq_idx[-env.exploration_total:]:
            a_gym = self.policy_hybrid(env, s0_line, s0_load, mode='test')
            s_gym, _, _, _ = env.step(a_gym)

            # update state
            s0_line = s_gym['line']
            s0_load = s_gym['load']
            # s0_pq = np.concatenate((s_gym['flow_p'], s_gym['flow_q']), axis=0)

        if self.if_timer == True:
            print("Complete testing using {}".format(time.time() - stime))

        if self.if_timer == True:
            print("Begin recording")
            stime = time.time()

        # ================ evaluate success =====================
        performance = sum(env.load_value_episode) - expert_load_profile_P[0] * len(expert.iter_time)  # we need to subtract the base load
        total_restored_load = sum(env.load_value_episode)

        if self.if_logging == True:
            self.logger.info("run test load status is {}".format(env.load_value_episode))
            if abs(goal) > 1e-4:
                self.logger.info("performance: {}; goal: {}".format(performance, goal))
            else:
                self.logger.info("nothing we can do to improve the load profile")

        # get disturbance index
        disturbance_idx = []
        for i in env.disturbance:
            id = int(i[i.find('_') + 1:])  # get the index name from the component name
            disturbance_idx.append(id)

        self.success_logger.add_score(performance, goal, total_restored_load, disturbance_idx, goal_1=goal_1)

        if self.if_timer == True:
            print("Complete recording using {}".format(time.time() - stime))

        return False


def main_hybrid_behavior_cloning(output_path):

    # ============= create GYM environment ===============
    env = gym.make(ENV_NAME_2, max_disturbance=MAX_DISTURBANCE, min_disturbance=MIN_DISTURBANCE)

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
        s0 = env.reset()

        if agent.if_logging == True:
            agent.logger.info("-------------------Run_train begin--------------------")
            agent.logger.info("The training disturbance is {}".format(env.disturbance))

        # get expert policy
        # note that expert only retrieve information from the environment but will not change it
        agent.get_expert_policy(env, s0)

        # determine if use expert advice or learned policy
        # main difference between behavior cloning and DAGGER
        if it > NUM_TRAIN_EPISODES:
            agent.warmstart = False

        # executes the expert policy and perform imitation learning
        agent.run_train(env, s0)

        # test current trained policy network using new environment from certain iterations
        if it >= 1:
            # initialize environment
            s0 = env.reset()
            # execute learned policy on the environment
            flag_convergence = agent.run_test(env, s0)

        if flag_convergence == True:
            break

        agent.total_episode = agent.total_episode + 1

    return agent




if __name__ == "__main__":

    # ========== run hierarchical_behavior_cloning===========
    output_path = os.getcwd()
    output_path = output_path + "/results/results_BC_hybrid_stochastic_dist/n_2_feature_load/"
    agent = main_hybrid_behavior_cloning(output_path)

    # ================== save trajectory =======================
    print('==================== saving trajectory =====================')
    trajectory_tieline = pd.DataFrame(columns=["state", 'action'])
    idx_not_none = [i for i, v in enumerate(agent.nn_tieline.replay_hist) if v != None]
    for i in idx_not_none:
        _state = agent.nn_tieline.replay_hist[i][0].tolist()
        _action = agent.nn_tieline.replay_hist[i][1].tolist()
        trajectory_tieline = trajectory_tieline.append({'state': _state, 'action': _action}, ignore_index=True)
    trajectory_tieline.to_csv(output_path + "trajectory_BC_hybrid_high" + agent.dt_string + '.csv')

    trajectory_var0 = pd.DataFrame(columns=["state", 'action'])
    idx_not_none = [i for i, v in enumerate(agent.nn_varcon[0].replay_hist) if v != None]
    for i in idx_not_none:
        _state = agent.nn_varcon[0].replay_hist[i][0].tolist()
        _action = agent.nn_varcon[0].replay_hist[i][1].tolist()
        trajectory_var0 = trajectory_var0.append({'state': _state, 'action': _action}, ignore_index=True)
    trajectory_var0.to_csv(output_path + "trajectory_BC_hybrid_var0" + agent.dt_string + '.csv')

    trajectory_var1 = pd.DataFrame(columns=["state", 'action'])
    idx_not_none = [i for i, v in enumerate(agent.nn_varcon[1].replay_hist) if v != None]
    for i in idx_not_none:
        _state = agent.nn_varcon[1].replay_hist[i][0].tolist()
        _action = agent.nn_varcon[1].replay_hist[i][1].tolist()
        trajectory_var1 = trajectory_var1.append({'state': _state, 'action': _action}, ignore_index=True)
    trajectory_var1.to_csv(output_path + "trajectory_BC_hybrid_var1" + agent.dt_string + '.csv')
    voltage_step_count = 0
    for i in trajectory_var1['action'].keys():
        if sum(trajectory_var1['action'][i]) >= 0.01:
            voltage_step_count = voltage_step_count + 1.0
    print("Var 1 Usage {}".format(voltage_step_count / NUM_TOTAL_EPISODES))

    trajectory_var2 = pd.DataFrame(columns=["state", 'action'])
    idx_not_none = [i for i, v in enumerate(agent.nn_varcon[2].replay_hist) if v != None]
    for i in idx_not_none:
        _state = agent.nn_varcon[2].replay_hist[i][0].tolist()
        _action = agent.nn_varcon[2].replay_hist[i][1].tolist()
        trajectory_var2 = trajectory_var2.append({'state': _state, 'action': _action}, ignore_index=True)
    trajectory_var2.to_csv(output_path + "trajectory_BC_hybrid_var2" + agent.dt_string + '.csv')
    voltage_step_count = 0
    for i in trajectory_var2['action'].keys():
        if sum(trajectory_var2['action'][i]) >= 0.01:
            voltage_step_count = voltage_step_count + 1.0
    print("Var 2 Usage {}".format(voltage_step_count / NUM_TOTAL_EPISODES))

    trajectory_var3 = pd.DataFrame(columns=["state", 'action'])
    idx_not_none = [i for i, v in enumerate(agent.nn_varcon[3].replay_hist) if v != None]
    for i in idx_not_none:
        _state = agent.nn_varcon[3].replay_hist[i][0].tolist()
        _action = agent.nn_varcon[3].replay_hist[i][1].tolist()
        trajectory_var3 = trajectory_var3.append({'state': _state, 'action': _action}, ignore_index=True)
    trajectory_var3.to_csv(output_path + "trajectory_BC_hybrid_var3" + agent.dt_string + '.csv')
    voltage_step_count = 0
    for i in trajectory_var3['action'].keys():
        if sum(trajectory_var3['action'][i]) >= 0.01:
            voltage_step_count = voltage_step_count + 1.0
    print("Var 3 Usage {}".format(voltage_step_count / NUM_TOTAL_EPISODES))

    trajectory_var4 = pd.DataFrame(columns=["state", 'action'])
    idx_not_none = [i for i, v in enumerate(agent.nn_varcon[4].replay_hist) if v != None]
    for i in idx_not_none:
        _state = agent.nn_varcon[4].replay_hist[i][0].tolist()
        _action = agent.nn_varcon[4].replay_hist[i][1].tolist()
        trajectory_var4 = trajectory_var4.append({'state': _state, 'action': _action}, ignore_index=True)
    trajectory_var4.to_csv(output_path + "trajectory_BC_hybrid_var4" + agent.dt_string + '.csv')
    voltage_step_count = 0
    for i in trajectory_var4['action'].keys():
        if sum(trajectory_var4['action'][i]) >= 0.01:
            voltage_step_count = voltage_step_count + 1.0
    print("Var 4 Usage {}".format(voltage_step_count / NUM_TOTAL_EPISODES))

    trajectory_var5 = pd.DataFrame(columns=["state", 'action'])
    idx_not_none = [i for i, v in enumerate(agent.nn_varcon[5].replay_hist) if v != None]
    for i in idx_not_none:
        _state = agent.nn_varcon[5].replay_hist[i][0].tolist()
        _action = agent.nn_varcon[5].replay_hist[i][1].tolist()
        trajectory_var5 = trajectory_var5.append({'state': _state, 'action': _action}, ignore_index=True)
    trajectory_var5.to_csv(output_path + "trajectory_BC_hybrid_var5" + agent.dt_string + '.csv')
    voltage_step_count = 0
    for i in trajectory_var5['action'].keys():
        if sum(trajectory_var5['action'][i]) >= 0.01:
            voltage_step_count = voltage_step_count + 1.0
    print("Var 5 Usage {}".format(voltage_step_count / NUM_TOTAL_EPISODES))

    # print('==================== saving nn policy =====================')
    # agent.nn_tieline.model.save("DNN_BC_hybrid_tieline" + agent.dt_string + '.h5')
    # agent.nn_varcon[0].model.save("DNN_BC_hybrid_var0" + agent.dt_string + '.h5')
    # agent.nn_varcon[1].model.save("DNN_BC_hybrid_var1" + agent.dt_string + '.h5')
    # agent.nn_varcon[2].model.save("DNN_BC_hybrid_var2" + agent.dt_string + '.h5')
    # agent.nn_varcon[3].model.save("DNN_BC_hybrid_var3" + agent.dt_string + '.h5')
    # agent.nn_varcon[4].model.save("DNN_BC_hybrid_var4" + agent.dt_string + '.h5')
    # agent.nn_varcon[5].model.save("DNN_BC_hybrid_var5" + agent.dt_string + '.h5')

    # # ========== test hierarchical_behavior_cloning===========
    # from keras.models import load_model
    # model = load_model("DNN_BC__2020_10_02_15_46.h5")
    # main_behavior_cloning_testing(output_path, model, "training")




