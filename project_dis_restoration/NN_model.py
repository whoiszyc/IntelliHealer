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
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Conv2D, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential
from keras.layers import Input
from keras.models import Model

# Parameters
ALPHA = 0.1
GAMMA = 0.95
LEARNING_RATE = 0.001
TRAIN_HIST_SIZE = 100000
EXPERIENCE_MEMORY = 100000
BATCH_SIZE = 32
MEMORY_SIZE = 10000000
EXPLORATION_MAX = 1.0
EXPLORATION_MIN = 0.01
EXPLORATION_DECAY = 0.999


class LossHistory(keras.callbacks.Callback):
	def on_train_begin(self, logs={}):
		self.losses = []

	def on_batch_end(self, batch, logs={}):
		self.losses.append(logs.get('loss'))


class CNN_TieLine:
    def __init__(self, observation_space, action_space):
        inputs = Input(shape=(33, 33, 1))
        x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = Conv2D(32, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        conv_out = Flatten()(x)
        y = Dense(512, activation='relu')(conv_out)
        action_outputs = Dense(6, activation='softmax', name='action_outputs')(y)
        self.model = Model(inputs=inputs, outputs=action_outputs)
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))  # categorical_crossentropy，  kullback_leibler_divergence
        self.output_shape = action_space
        self.input_shape = observation_space
        self.ind = 0  # sample index
        self.replay_hist = [None] * TRAIN_HIST_SIZE
        self._stats_loss = []
        self._history = LossHistory()

    def collect(self, s, expert_a):
        # "replay_hist" has been added through "collect" function
        # one-hot encode action
        a = [0] * self.output_shape
        a[expert_a] = 1
        a = np.array(a)
        s = np.array(s)
        if s is not None:
            # "processed: is the system state -- input to the neural net
            # "replay_hist" stores the state-action pair (state, action)
            self.replay_hist[self.ind] = (s.astype(np.float32), a.astype(np.float32))

            # if the collection is lager than TRAIN_HIST_SIZE, the collection will start from the beginning
            self.ind = (self.ind + 1) % TRAIN_HIST_SIZE

    def end_collect(self):
        # training is performed using "end_collect" function
        try:
            return self.train()
        except:
            return

    def train(self):
        # if not reached TRAIN_HIST_SIZE yet, then get the number of samples
        self._num_valid = self.ind if self.replay_hist[-1] == None else TRAIN_HIST_SIZE

        try:
            self._samples = range(self._num_valid)
            BATCH_SIZE = len(self._samples)
        except:
            self._samples = range(self._num_valid) + [0] * (BATCH_SIZE - len(range(self._num_valid)))

        # convert replay data to trainable data
        self._selected_replay_data = [self.replay_hist[i] for i in self._samples]
        self._train_x = np.reshape([self._selected_replay_data[i][0] for i in range(BATCH_SIZE)],
                                   (BATCH_SIZE, 33, 33, 1))
        self._train_y = np.reshape([self._selected_replay_data[i][1] for i in range(BATCH_SIZE)], (BATCH_SIZE, self.output_shape))

        # start training
        self.model.fit(self._train_x, self._train_y, batch_size=32, verbose=0, epochs=1, callbacks=[self._history])
        return self._history.losses

    def predict(self, x, batch_size=1):
        """predict on (a batch of) x"""
        return self.model.predict(x, batch_size=batch_size, verbose=0)



class DNN_TieLine:
    def __init__(self, observation_space, action_space):
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(observation_space,), activation="relu"))  # sigmoid    relu
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(action_space, activation="softmax"))
        self.model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=0.001))  # categorical_crossentropy，  kullback_leibler_divergence
        self.output_shape = action_space
        self.input_shape = observation_space
        self.ind = 0  # sample index
        self.replay_hist = [None] * TRAIN_HIST_SIZE
        self.replay_hist_other = [None] * TRAIN_HIST_SIZE  # record other status for GNN features
        self.memory = deque(maxlen=MEMORY_SIZE)
        self._stats_loss = []
        self._history = LossHistory()

    def collect(self, s, expert_a, other=None):
        # "replay_hist" has been added through "collect" function
        # one-hot encode action
        a = [0] * self.output_shape
        a[expert_a] = 1
        a = np.array(a)
        s = np.array(s)

        # "processed: is the system state -- input to the neural net
        # "replay_hist" stores the state-action pair (state, action)
        self.replay_hist[self.ind] = (s.astype(np.float32), a.astype(np.float32))

        # record other system status
        if other is not None:
            self.replay_hist_other[self.ind] = other

        # if the collection is lager than TRAIN_HIST_SIZE, the collection will start from the beginning
        self.ind = (self.ind + 1) % TRAIN_HIST_SIZE

    def end_collect(self):
        # training is performed using "end_collect" function
        try:
            return self.train()
        except:
            return

    def train(self):
        # if not reached TRAIN_HIST_SIZE yet, then get the number of samples
        self._num_valid = self.ind if self.replay_hist[-1] == None else TRAIN_HIST_SIZE

        try:
            self._samples = range(self._num_valid)
            BATCH_SIZE = len(self._samples)
        except:
            self._samples = range(self._num_valid) + [0] * (BATCH_SIZE - len(range(self._num_valid)))

        # convert replay data to trainable data
        self._selected_replay_data = [self.replay_hist[i] for i in self._samples]
        self._train_x = np.reshape([self._selected_replay_data[i][0] for i in range(BATCH_SIZE)],
                                   (BATCH_SIZE, self.input_shape))
        self._train_y = np.reshape([self._selected_replay_data[i][1] for i in range(BATCH_SIZE)], (BATCH_SIZE, self.output_shape))

        # start training
        self.model.fit(self._train_x, self._train_y, batch_size=32, verbose=0, epochs=1, callbacks=[self._history])
        return self._history.losses

    def predict(self, x, batch_size=1):
        """predict on (a batch of) x"""
        return self.model.predict(x, batch_size=batch_size, verbose=0)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def experience_replay(self):
        if len(self.memory) < BATCH_SIZE:
            return
        batch = random.sample(self.memory, BATCH_SIZE)
        for state, action, reward, state_next, terminal in batch:
            q_update = reward
            if not terminal:
                q_update = (reward + GAMMA * np.amax(self.model.predict(state_next)[0]))
            q_values = self.model.predict(state)
            q_values[0][action] = q_update
            self.model.fit(state, q_values, verbose=0)


class DNN_VarCon:
    def __init__(self, observation_space, action_space):
        self.model = Sequential()
        self.model.add(Dense(64, input_shape=(observation_space,), activation="relu"))  # sigmoid  tanh  relu
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(action_space, activation="tanh"))
        self.model.compile(loss="mean_squared_error", optimizer=Adam(lr=0.001))  # categorical_crossentropy，  kullback_leibler_divergence
        self.output_shape = action_space
        self.input_shape = observation_space
        self.ind = 0  # sample index
        self.replay_hist = [None] * TRAIN_HIST_SIZE
        self._stats_loss = []
        self._history = LossHistory()

    def collect(self, s, expert_a):
        # "replay_hist" has been added through "collect" function
        a = np.array(expert_a)
        s = np.array(s)
        if s is not None:
            # "processed: is the system state -- input to the neural net
            # "replay_hist" stores the state-action pair (state, action)
            self.replay_hist[self.ind] = (s.astype(np.float32), a.astype(np.float32))

            # if the collection is lager than TRAIN_HIST_SIZE, the collection will start from the beginning
            self.ind = (self.ind + 1) % TRAIN_HIST_SIZE

    def end_collect(self):
        # training is performed using "end_collect" function
        try:
            return self.train()
        except:
            return

    def train(self):
        # if not reached TRAIN_HIST_SIZE yet, then get the number of samples
        self._num_valid = self.ind if self.replay_hist[-1] == None else TRAIN_HIST_SIZE

        try:
            self._samples = range(self._num_valid)
            BATCH_SIZE = len(self._samples)
        except:
            self._samples = range(self._num_valid) + [0] * (BATCH_SIZE - len(range(self._num_valid)))

        # convert replay data to trainable data
        self._selected_replay_data = [self.replay_hist[i] for i in self._samples]
        self._train_x = np.reshape([self._selected_replay_data[i][0] for i in range(BATCH_SIZE)],
                                   (BATCH_SIZE, self.input_shape))
        self._train_y = np.reshape([self._selected_replay_data[i][1] for i in range(BATCH_SIZE)], (BATCH_SIZE, self.output_shape))

        # start training
        self.model.fit(self._train_x, self._train_y, batch_size=32, verbose=0, epochs=1, callbacks=[self._history])
        return self._history.losses

    def predict(self, x, batch_size=1):
        """predict on (a batch of) x"""
        return self.model.predict(x, batch_size=batch_size, verbose=0)
