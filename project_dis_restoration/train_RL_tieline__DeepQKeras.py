import time
from datetime import datetime
import os
import sys
import logging
import random
import gym
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
WARM_START_OFF = 10000
NUM_TOTAL_EPISODES = 10000

ALPHA = 0.1
GAMMA = 0.95
LEARNING_RATE = 0.001
MEMORY_SIZE = 1000000
BATCH_SIZE = 50
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 1   # The exploration decay highly depends on the episode number
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

        self.exploration_rate = EXPLORATION_MAX

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


    def policy(self, s, env):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(env.action_space.n)
        q_values = self.nn_tieline.predict(s)
        return np.argmax(q_values[0])


    def update_rate(self):
        self.exploration_rate *= EXPLORATION_DECAY
        self.exploration_rate = max(EXPLORATION_MIN, self.exploration_rate)


    def run_train(self, env, s0):

        if self.if_timer == True:
            print("Begin exploration")
            stime = time.time()

        if self.if_logging == True:
            self.logger.info("-------------------Run_train begin--------------------")
            self.logger.info("The training disturbance is {}".format(env.disturbance))

        s0 = np.reshape(s0, [1, self.nn_tieline.input_shape])

        for step in env.exploration_seq_idx[-env.exploration_total:]:
            a = self.policy(s0, env)

            # random action also works, but slower since you can bump into walls
            # a = np.random.choice(ALL_POSSIBLE_ACTIONS)
            s, r, done, _ = env.step(a)
            s = np.reshape(s, [1, self.nn_tieline.input_shape])
            r = r if not done else -r
            # collect data
            self.nn_tieline.remember(s0, a, r, s, done)
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
            self.nn_tieline.experience_replay()

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
        ppc = case33_tieline()
        expert = OutageManage()
        expert.data_preparation(ppc, env.disturbance, 1.05, 0.05)
        expert.solve_network_restoration(len(env.exploration_seq_idx), env.state_line_status, env.state_load_status)
        opt = pm.SolverFactory("cplex", executable='/Applications/CPLEX_Studio128/cplex/bin/x86-64_osx/cplex')
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
                    self.logger.info("learning given state {} at step {}".format(s0[0][-5:], step))
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
                    self.logger.info("learning given state {} at step {}".format(s0[0][-5:], step))
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



def main_DQN(output_path):
    # ============= create GYM environment ===============
    env = gym.make(ENV_NAME_1, max_disturbance=MAX_DISTURBANCE, min_disturbance=MIN_DISTURBANCE)

    # ============== create agent ===================
    agent = Agent(env, output_path)

    # ============= Begin main training loop ===========
    flag_convergence = False  # set convergence flag to be false
    tic = time.perf_counter()  # start clock
    for it in range(NUM_TOTAL_EPISODES):
        if it % 1 == 0:
            toc = time.perf_counter()
            print("===================================================")
            print(f"Training time: {toc - tic:0.4f} seconds; Mission {it:d} of {NUM_TOTAL_EPISODES:d}")
            print("===================================================")
        agent.logger.info(f"=============== Mission {it:d} of {NUM_TOTAL_EPISODES:d} =================")

        # initialize environment for training
        s0, _ = env.reset()
        # executes the expert policy and perform Deep Q learning
        agent.run_train(env, s0)

        # initialize environment for testing
        s0, _ = env.reset()
        # execute learned policy on the environment
        flag_convergence = agent.run_test(env, s0)

        if flag_convergence == True:
            break

        agent.total_episode = agent.total_episode + 1

    return agent


if __name__ == "__main__":

    output_path = os.getcwd()
    output_path = output_path + "/results/results_DQN_tieline_stochastic_dist/n_5/"

    # # ========== train hierarchical_behavior_cloning===========
    agent = main_DQN(output_path)


