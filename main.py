import gym
import numpy as np
import os.path
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense

TRAINING_EPISODES = 10000

class Memory:
    def __init__(self, size):
        self.memory = deque(maxlen=size)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def add(self, sample):
        self.memory.append(sample)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.memory = Memory(10000)
        self.model = self._build()

    def _build(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='adam')
        if os.path.isfile('./save/model.h5'):
            model.load_weights('./save/model.h5')
        return model

    def _save(self):
        self.model.save('./save/model.h5')

    def observe(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state))

    def learn(self, batch_size):
        if len((self.memory)) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in batch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
                target_f = self.model.predict(state)
                target_f[0][action] = target
                self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


env = gym.make('CartPole-v0')
env.reset()
for i_episode in range(TRAINING_EPISODES):
    observation = env.reset()
    while True:
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            break
