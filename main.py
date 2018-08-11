import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

def preprocess(I):
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1
    return I.astype(np.float).ravel()

def get_epsilon(e):
    return max(EPSILON_MIN, min(EPSILON, 1.0 - math.log10((e + 1) * EPSILON_DECAY)))

def build_model():
    model = Sequential()
    model.add(Dense(200, input_shape=(STATE_SIZE,), activation='relu'))
    model.add(Dense(40, activation='relu'))
    model.add(Dense(ACTION_SIZE, activation='linear'))
    model.compile(loss='mse',
                  optimizer=Adam(lr=LEARNING_RATE))
    return model

def remember(state, action, reward, next_state, done):
    MEMORY.append((state, action, reward, next_state, done))

def predict(state):
    state = np.reshape(state, [1, STATE_SIZE])
    return model.predict(state)

def act(state):
    if np.random.rand() <= EPSILON:
        return random.randrange(ACTION_SIZE)
    act_values = predict(state)
    return np.argmax(act_values[0])  # returns action

def replay(batch_size):
    minibatch = random.sample(MEMORY, batch_size)
    for state, action, reward, next_state, done in minibatch:
        target = reward
        if not done:
            target = (reward + GAMMA *
                      np.amax(predict(next_state)[0]))
        target_f = predict(state)
        target_f[0][action] = target

        state = np.reshape(state, [1, STATE_SIZE])
        model.fit(state, target_f, epochs=1, verbose=0)


env = gym.make('Pong-v0')

EPISODES = 6000
LEARNING_RATE = 1e-4
BATCH_SIZE = 1000

STATE_SIZE = 80 * 80
ACTION_SIZE = env.action_space.n
MEMORY = deque(maxlen=10000)
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY=0.995

model = build_model()


if __name__ == "__main__":
    # Monitoring
    running_avg = deque(maxlen=100)

    for e in range(EPISODES):
        state = env.reset()
        prev = None
        R = 0
        done = False
        while not done:
            env.render()

            curr = preprocess(state)
            state = curr - prev if prev is not None else np.zeros(STATE_SIZE)
            prev = curr

            action = act(state)
            state_, reward, done, _ = env.step(action)
            curr_ = preprocess(state_)
            next_state = curr_ - prev
            remember(state, action, reward, next_state, done)
            state = state_
            print(reward)

            R += reward

            if done:
                running_avg.append(R)
                avg = round(sum(running_avg) / len(running_avg))
                print("episode: {}/{}, score: {}, e: {:.2}, avg (100 eps): {}"
                      .format(e, EPISODES, R, EPSILON, avg))
                EPSILON = get_epsilon(e)
                break
        if (e % 10 == 0 and e > 0):
            replay(BATCH_SIZE)
