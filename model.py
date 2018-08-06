from keras.models. import Sequential
from keras.layers import Dense

class Model:
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

    def build(self):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=self.num_states))
        model.add(Dense(self.num_actions, activation='linear'))
        model.compile(loss='mse', optimizer='rmsprop')
        self.model = model

    def train(self):
        
