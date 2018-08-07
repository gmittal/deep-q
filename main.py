import gym
import numpy as np
import random
import scipy
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
from keras import backend as K

EPISODES = 5000

def huber_loss(target, prediction):
    return K.mean(K.sqrt(1+K.square(prediction-target))-1, axis=-1)

def preprocess_state(state):
    return np.reshape(state, [1, 4])

class Memory:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, sample):
        self.buffer.append(sample)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class Agent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = Memory(2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(units=self.action_size, activation='linear'))
        model.compile(loss=huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.add((state, action, reward, next_state, done))

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

if __name__ == "__main__":
    env = gym.make('CartPole-v1')
    env._max_episode_steps = None
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)

    done = False
    batch_size = 32

    frames = 0

    for e in range(EPISODES):
        state = env.reset()
        state = preprocess_state(state)

        total_reward = 0

        while True:
            frames += 1
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = preprocess_state(next_state)
            agent.remember(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            if frames > 100:
                frames = 0
                agent.update_target_model()

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, total_reward, agent.epsilon))
                break
