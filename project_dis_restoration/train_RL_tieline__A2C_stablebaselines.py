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
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from stable_baselines3 import A2C


# Parameters
ENV_NAME_1 = "RestorationDisEnv-v0"
ENV_NAME_2 = "RestorationDisEnv-v1"
MAX_DISTURBANCE = 1
MIN_DISTURBANCE = 1

env = gym.make('RestorationDisEnv-v1', max_disturbance=MAX_DISTURBANCE, min_disturbance=MIN_DISTURBANCE)
model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=1000, log_interval=4)
model.save("Restoration_dqn.pkl")


