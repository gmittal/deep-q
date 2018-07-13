import gym
import math
import numpy as np
import random
from collections import deque
from keras.models import Sequential
from keras.layers import Dense

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        self.memory = deque(maxlen=100000)
        self.model = self.build()

        self.gamma = 0.99
        self.epsilon = 1
        self.min_epsilon = 0.01
        self.decay = 0.001

        self.steps = 0

    def build(self):
        model = Sequential()
        model.add(Dense(output_dim=64, activation='relu', input_dim=self.state_size))
        model.add(Dense(output_dim=self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer='rmsprop')
        return model

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size-1)
        else:
            return np.argmax(self.model.predict(state.reshape(1, self.state_size)).flatten())

    def observe(self, sample):
        self.memory.append(sample)

        # Epsilon reduction
        self.steps += 1
        self.epsilon = self.min_epsilon + (1 - self.min_epsilon) * math.exp(-self.decay * self.steps)

    def replay(self):
        if (self.steps < 64):
            return

        batch = random.sample(self.memory, 64)

        no_state = np.zeros(self.state_size)

        states = np.array([o[0] for o in batch])
        next_states = np.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.model.predict(states)
        p_ = self.model.predict(next_states)

        x = np.zeros((len(batch), self.state_size))
        y = np.zeros((len(batch), self.action_size))

        for i in range(len(batch)):
            o = batch[i]
            state = o[0]; action = o[1]; reward = o[2]; next_state = o[3]

            target = p[i]
            if next_state is None:
                target[action] = reward
            else:
                target[action] = reward + self.gamma * np.amax(p_[i])

            x[i] = state
            y[i] = target

        self.model.fit(x, y, batch_size=64, epochs=1, verbose=0)

# Runs a single episode
def run(env, agent):
    state = env.reset()
    R = 0

    while True:
        env.render()
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)

        if done:
            next_state = None

        agent.observe((state, action, reward, next_state))
        agent.replay()

        state = next_state
        R += reward

        if done:
            break

    print("Total reward: ", R)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    agent = Agent(env.observation_space.shape[0], env.action_space.n)
    for episode in range(5000):
        run(env, agent)
