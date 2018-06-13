from collections import deque
import gym
import gym_super_mario_bros
import numpy as np
import random
import tensorflow as tf
from collections import deque
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D

env = gym_super_mario_bros.make('SuperMarioBros-v0')

GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
EPISODES = 10

memory = deque(maxlen=2000)

# Build the deep model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=env.observation_space.shape))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(state):
    if np.random.rand() <= EPSILON:
        return env.action_space.sample()
    return np.argmax(model.predict(state)[0])

def replay(batch_size):
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = reward + GAMMA * np.amax(model.predict(next_state)[0])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1)
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY


for episode in range(100):
    state = env.reset()

    for t in range(500):
        action = act(state)
        next_state, reward, done, _ = env.step(action)
        remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}".format(e, 100, t))
            break

    replay(32)
