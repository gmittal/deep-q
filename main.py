from collections import deque
import gym
import numpy as np
import random
import tensorflow as tf
from collections import deque
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D

env = gym.make('CartPole-v0')

GAMMA = 0.95
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
EPISODES = 10

memory = deque(maxlen=10000)

# Build the deep model
model = Sequential()
model.add(Dense(30, activation='relu', input_dim=4))
model.add(Dense(128, activation='relu'))
model.add(Dense(env.action_space.n, activation='linear'))
model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

def remember(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))

def act(state):
    state = np.reshape(state, [1, 4])
    if np.random.rand() <= EPSILON:
        return env.action_space.sample()
    return np.argmax(model.predict(state)[0])

def replay(batch_size):
    global EPSILON
    minibatch = random.sample(memory, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        next_state = np.reshape(next_state, [1, 4])
        if not done:
            target = reward + GAMMA * np.amax(model.predict(next_state)[0])
        state = np.reshape(state, [1, 4])
        target_f = model.predict(state)
        target_f[0][action] = target
        model.fit(state, target_f, epochs=1, verbose=0)
    if EPSILON > EPSILON_MIN:
        EPSILON *= EPSILON_DECAY

for episode in range(1000):
    state = env.reset()
    env.render()

    for t in range(200):
        action = act(state)
        next_state, reward, done, _ = env.step(action)
        remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            print("episode: {}/{}, score: {}".format(episode, 1000, t))
            break

    replay(12)


# Test it out!
s = env.reset()
done = False
episode_reward = 0
while not done:
    env.render()
    a = act(s)
    s2, r, done, info = env.step(a)
    episode_reward += r

print(f"Episode Reward: {episode_reward}")
