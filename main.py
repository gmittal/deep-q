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

def process_image(img):
    rgb = scipy.misc.imresize(img, (84, 84), interp='bilinear')
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b     # extract luminance
    o = gray.astype('uint8') / 128 - 1    # normalize
    return o

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
        model.add(Conv2D(32, (8, 8), strides=(4,4), activation='relu', input_shape=(self.state_size), data_format='channels_first'))
        model.add(Conv2D(64, (4, 4), strides=(2,2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(units=512, activation='relu'))

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
        act_values = self.model.predict(state.reshape(1, 2, 84, 84)).flatten()
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = self.memory.sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state.reshape(1, 2, 84, 84))
            if done:
                target[0][action] = reward
            else:
                # a = self.model.predict(next_state)[0]
                t = self.target_model.predict(next_state.reshape(1, 2, 84, 84))[0]
                target[0][action] = reward + self.gamma * np.amax(t)
                # target[0][action] = reward + self.gamma * t[np.argmax(a)]
            self.model.fit(state.reshape(1, 2, 84, 84), target, epochs=1, verbose=0)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = gym.make('Pong-v0')
    env._max_episode_steps = None
    state_size = (2, 84, 84)
    action_size = env.action_space.n
    agent = Agent(state_size, action_size)

    done = False
    batch_size = 32

    frames = 0

    for e in range(EPISODES):
        img = env.reset()
        w = process_image(img)
        state = np.array([w, w])
        print(state.shape)

        total_reward = 0

        while True:
            frames += 1
            # env.render()
            action = agent.act(state)
            img, reward, done, _ = env.step(action)
            next_state = np.array([state[1], process_image(img)])
            reward = np.clip(reward, -1, 1)   # clip reward to [-1, 1]

            agent.remember(state, action, reward, next_state, done)
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            total_reward += reward

            if frames > 10:
                frames = 0
                agent.update_target_model()

            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, EPISODES, total_reward, agent.epsilon))
                break
