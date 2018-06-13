from collections import deque
import gym
import gym_super_mario_bros
import numpy as np
import random
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D

env = gym_super_mario_bros.make('SuperMarioBros-v0')
observation = env.reset() / 255
print(observation)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=observation.shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(env.action_space.n, init='uniform', acvitation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
