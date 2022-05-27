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
from gym_power_res.envs.data_test_case import case33_tieline, case33_tieline_DG, case119_tieline


# Parameters
ENV_NAME_1 = "RestorationDisEnv-v0"
ENV_NAME_2 = "RestorationDisVarConEnv-v0"
ENV_NAME_3 = "RestorationDisEnv119-v0"

# WARM_START_OFF and NUM_TOTAL_EPISODES can be different only for DAGGER algorithms
WARM_START_OFF = 1000
NUM_TOTAL_EPISODES = 1000
MAX_DISTURBANCE = 1
MIN_DISTURBANCE = 1

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
        try:
            num_varcon = len(env.ppc['varcon'])
        except:
            pass

        self.nn_tieline = DNN_TieLine(env.observation_space.shape[0], env.action_space.n)

        self.total_episode = 0

        self.warmstart = True

        self.if_logging = False
        
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
        self.success_logger = SuccessLogger(ENV_NAME_1, output_path, title='Behavior Cloning')  # log for performance evaluation


    def get_expert_policy(self, env, s0):
        """
        get the expert policy by solving the full-period restoration problem
        """
        if self.if_timer == True:
            print("Begin quering expert")
            stime = time.time()

        expert = OutageManage()
        expert.data_preparation(env.ppc, env.disturbance)
        expert.solve_network_restoration(len(env.exploration_seq_idx), env.state_line_status, env.state_load_status)
        opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
        # opt = pm.SolverFactory("cplex")
        opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
        expert_result = opt.solve(expert.model, tee=False)

        # if optimal, retrieve the expert actions
#         if expert_result['solver'][0]['Termination condition'].key == 'optimal':
        if expert_result.solver.termination_condition == TerminationCondition.optimal:
            load_status = expert.get_solution_2d('rho', expert.iter_bus, expert.iter_time)
            line_status = expert.get_solution_2d('ul', expert.iter_line, expert.iter_time)
            # get load profile
            expert_load_profile_bin, expert_load_profile_P = self.load_profile(env, load_status)
            if self.if_logging == True:
                self.logger.info("Expert return optimal solution: {}".format(expert_load_profile_bin))
                self.logger.info("Expert return optimal solution: {}".format(expert_load_profile_P))
        else:
            # TODO: what to do when the expert problem is infeasible
            self.logger.info("expert does not return optimal solution under disturbance {}".format(env.disturbance))
            print("expert is not able to provide action under disturbance {}".format(env.disturbance))

        if self.if_timer == True: 
            print("Complete quering expert using {}".format(time.time() - stime))

        # store (state, action) pairs
        # s0 -> a1 -> s1 -> a2 -> s2 -> a3 -> s3 -> a4 -> s4 -> a5 -> s5
        # in this case, we need to store (s0, a1), (s1, a2), (s2, a3), (s3, a4), (s4, a5)
        self.expert = DictList()  # since we may have the same state, we will store expert action in new DictList object

        for step in env.exploration_seq_idx[-env.exploration_total:]:
            s0 = tuple(s0)
            s0_tieline = s0[-15:]  # TODO: number of tielies
            # get action w.r.t. s0 by minus tieline status
            s1_tieline = []
            for i in env.ppc["tieline"]:
                s1_tieline.append(round(line_status[i][step]))
            # minus the list to get the action
            a1 = self.convert_action(s1_tieline, s0_tieline)

            # record to logger
            if self.if_logging == True:
                self.logger.info("expert state at step {} is {}".format(step, s0))
                self.logger.info("expert action at step {} is {}".format(step, a1))

            # store state-action pair as DictList
            if s0 not in self.expert:
                self.expert[s0] = []
                self.expert[s0].append(a1)
            else:
                self.expert[s0].append(a1)

            # update state under action a1
            s0 = []
            for i in env.ppc["iter_line"]:
                s0.append(round(line_status[i][step]))


    def load_profile(self, env, load_status):
        load_profile_bin = []
        load_profile_P = []
        total_step = len(load_status['bus_1'])
        for t in range(total_step):
            _tem_bin = []
            _tem_P = []
            for i in load_status:
                id = int(i[i.find('_') + 1:])  # get the matrix index from the component name
                idx = np.where(id == env.ppc['bus'])[0][0]
                _tem_bin.append(load_status[i][t])
                _tem_P.append(load_status[i][t] * env.ppc['bus'][idx, 4])
            load_profile_bin.append(sum(_tem_bin))
            load_profile_P.append(sum(_tem_P))
        return load_profile_bin, load_profile_P


    def convert_action(self, s1_tieline, s0_tieline):
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
        if d == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            a = 0  # no action
        elif d == [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            a = 1
        elif d == [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            a = 2
        elif d == [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            a = 3
        elif d == [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            a = 4
        elif d == [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            a = 5
        elif d == [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]:
            a = 6
        elif d == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]:
            a = 7
        elif d == [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]:
            a = 8
        elif d == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]:
            a = 9
        elif d == [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]:
            a = 10
        elif d == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]:
            a = 11
        elif d == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]:
            a = 12
        elif d == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]:
            a = 13
        elif d == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]:
            a = 14
        elif d == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]:
            a = 15
        else:
            print("Multiple actions exist in one step!")
            sys.exit()
        return a


    def policy(self, s):
        if self.warmstart:  # if warm start is true, we use expert action
            action = self.expert[s][0]  # in case multiple actions have the same state
            return action
        else:               # if warm start is false, we use trained policy approximator
            s = np.reshape(s, (-1, self.nn_tieline.input_shape))
            action_prob = self.nn_tieline.predict(s)[0]
            return action_prob.argmax()


    def run_train(self, env, s0, l0=None):

        if self.if_timer == True:
            print("Begin exploration")
            stime = time.time()

        if self.if_logging == True:
            self.logger.info("-------------------Run_train begin--------------------")
            self.logger.info("The training disturbance is {}".format(env.disturbance))

        for step in env.exploration_seq_idx[-env.exploration_total:]:
            s0 = tuple(s0)
            a = self.policy(s0)

            # random action also works, but slower since you can bump into walls
            # a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            s, _, _, l = env.step(a)
            s = tuple(s)

            # collect (state,action) pair
            if l0 is not None:
                self.nn_tieline.collect(s0, a, l0)
            else:
                self.nn_tieline.collect(s0, a)
            if self.if_logging == True:
                self.logger.info("collected obs state {} at step {}".format(s0, step))
                self.logger.info("collected action {} at step {}".format(a, step))
                self.logger.info("next state {} at step {}".format(s, step))


            # update state
            s0 = s

        if self.if_timer == True:
            print("Complete exploration using {}".format(time.time() - stime))

        if self.if_timer == True:
            print("Begin training")
            stime = time.time()

        if self.total_episode >= 1:
            loss = self.nn_tieline.end_collect()
            self.nn_tieline._stats_loss.append(sum(loss) / len(loss))

        if self.if_timer == True:
            print("Complete training using {}".format(time.time() - stime))


    def run_test(self, env, s0, learned_model=None):

        if self.if_timer == True:
            print("Begin solving benchmark")
            stime = time.time()

        # ================ calculate Benchmark value to normalize the restoration as ratio ==================
        self.logger.info("-------------------Run_test begin--------------------")
        self.logger.info("The testing disturbance is {}".format(env.disturbance))
        # as a training benchmark, we only use tieline, which is the same action space with this agent
        ppc = case119_tieline()
        expert = OutageManage()
        expert.data_preparation(ppc, env.disturbance, 1.05, 0.05)
        expert.solve_network_restoration(len(env.exploration_seq_idx), env.state_line_status, env.state_load_status)
        opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')  #
#             opt = pm.SolverFactory("cplex")
        opt.options['mipgap'] = 0  # if gap=b, then it is (b*100) %
        expert_result = opt.solve(expert.model, tee=False)
        # if optimal, get the optimal value
#             if expert_result['solver'][0]['Termination condition'].key == 'optimal':
        if expert_result.solver.termination_condition == TerminationCondition.optimal:
            expert_sol_load_status = expert.get_solution_2d('rho', expert.iter_bus, expert.iter_time)
            expert_load_profile_bin, expert_load_profile_P = self.load_profile(env, expert_sol_load_status)
            if self.if_logging == True:
                self.logger.info("Benchmark return optimal solution: {}".format(expert_load_profile_bin))
                self.logger.info("Benchmark return optimal solution: {}".format(expert_load_profile_P))
            goal = sum(expert_load_profile_P) - expert_load_profile_P[0] * len(expert.iter_time)  # we need to subtract the base load
        else:
            self.logger.info("This scenario is infeasible; continue to the next training loop")
            Flag_continue = True

        if self.if_timer == True:
            print("Complete solving benchmark using {}".format(time.time() - stime))

        if self.if_timer == True:
            print("Begin testing")
            stime = time.time()

        # ============== run agent using trained policy approximator ==================
        if learned_model is None:
            for step in env.exploration_seq_idx[-env.exploration_total:]:
                s0 = np.reshape(s0, (-1, self.nn_tieline.input_shape))
                action_prob = self.nn_tieline.predict(s0)[0]  # this action is one-hot encoded
                a = action_prob.argmax()
                if self.if_logging == True:
                    self.logger.info("learning given state {} at step {}".format(s0[0][-15:], step)) #TODO
                    self.logger.info("learning action at step {} is {}".format(step, a))
                s, _, _, _ = env.step(a)
                s = tuple(s)

                # update state
                s0 = s
        else:
            for step in env.exploration_seq_idx[-env.exploration_total:]:
                s0 = np.reshape(s0, (-1, self.nn_tieline.input_shape))
                action_prob = learned_model.predict(s0)[0]  # this action is one-hot encoded
                a = action_prob.argmax()
                if self.if_logging == True:
                    self.logger.info("learning given state {} at step {}".format(s0[0][-15:], step)) #TODO
                    self.logger.info("learning action at step {} is {}".format(step, a))
                s, _, _, _ = env.step(a)
                s = tuple(s)

                # update state
                s0 = s
        
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
        
        self.success_logger.add_score(performance, goal, total_restored_load, disturbance_idx)

        if self.if_timer == True:
            print("Complete recording using {}".format(time.time() - stime))

        return False



def main_behavior_cloning(output_path):

    # ============= create GYM environment ===============
    env = gym.make(ENV_NAME_3, max_disturbance=MAX_DISTURBANCE, min_disturbance=MIN_DISTURBANCE)

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



if __name__ == "__main__":

    output_path = os.getcwd()
    output_path = output_path + "/results/results_BC_tieline_stochastic_dist_119/n_1/test/"

    # # ========== train hierarchical_behavior_cloning===========
    agent = main_behavior_cloning(output_path)

    ## ================== save trajectory =======================
    print('==================== saving trajectory =====================')
    trajectory = pd.DataFrame(columns=["line", "load", 'action'])
    for i in range(NUM_TOTAL_EPISODES * 5):
        _line = agent.nn_tieline.replay_hist[i][0].tolist()
        _load = agent.nn_tieline.replay_hist_other[i].tolist()
        _action = agent.nn_tieline.replay_hist[i][1].tolist()
        trajectory = trajectory.append({'line': _line, 'load': _load, 'action': _action}, ignore_index=True)
    trajectory.to_csv(output_path + "trajectory_BC" + agent.dt_string + '.csv')
    print('==================== trajectory is saved =====================')

    print('==================== saving nn policy =====================')
    agent.nn_tieline.model.save(output_path + "DNN_BC" + agent.dt_string + '.h5')
    print('==================== policy is saved =====================')
