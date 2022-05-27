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

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from stable_baselines3 import A2C

# Parameters
ENV_NAME_1 = "RestorationDisEnv-v0"
ENV_NAME_2 = "RestorationDisEnv-v1"
WARM_START_OFF = 200
NUM_TOTAL_EPISODES = 200

ALPHA = 0.1
GAMMA = 0.95
LEARNING_RATE = 0.0001  # 0.0007 is the default value
MEMORY_SIZE = 1000000
BATCH_SIZE = 50
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 1   # The exploration decay highly depends on the episode number
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



class Agent:
    def __init__(self, env, output_path, log_level=logging.DEBUG):

        self.model = A2C('MlpPolicy', env, learning_rate=LEARNING_RATE, verbose=1)

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
        self.success_logger = SuccessLogger(ENV_NAME_2, output_path, title='Behavior Cloning')  # log for performance evaluation


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
                s0 = np.reshape(s0, (-1, 37))
                a, _ = self.model.predict(s0)  # this action is one-hot encoded
                if self.if_logging == True:
                    self.logger.info("learning given state {} at step {}".format(s0[0][-5:], step))
                    self.logger.info("learning action at step {} is {}".format(step, a))
                s, _, _, _ = env.step(a[0])
                s = tuple(s)
                # update state
                s0 = s

        else:
            for step in env.exploration_seq_idx[-env.exploration_total:]:
                s0 = np.reshape(s0, (-1, 37))
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



def main_A2C(output_path):
    # ============= create GYM environment ===============
    env = gym.make('RestorationDisEnv-v1', max_disturbance=MAX_DISTURBANCE, min_disturbance=MIN_DISTURBANCE)

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

        # executes the expert policy and perform Deep Q learning
        agent.model.learn(total_timesteps=5)

        # initialize environment for testing
        s0 = env.reset()
        # execute learned policy on the environment
        flag_convergence = agent.run_test(env, s0)

        if flag_convergence == True:
            break

        agent.total_episode = agent.total_episode + 1

    return agent


if __name__ == "__main__":

    output_path = os.getcwd()
    output_path = output_path + "/results/results_AC_tieline_stochastic_dist/n_1/"

    # # ========== train A2C===========
    agent = main_A2C(output_path)


