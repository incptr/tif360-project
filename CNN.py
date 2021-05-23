import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np
from keras.models import Sequential, load_model
from abc import ABC, abstractmethod
import os

save_dir = '../saved_models'


class Network(ABC):
    def __init__(self, model):
        self.model = model
        self.load()

    def eval_position(self, board_state):
        input = self._board_states_to_inputs([board_state])
        result = self.model.predict(input)
        return result

    def update(self, board_states, rewards):
        inputs = self._board_states_to_inputs(board_states)
        outputs = np.array(rewards)
        self.model.train_on_batch(inputs, outputs)

    def load(self):
        model_path = os.path.join(save_dir, self.get_save_file())
        if os.path.isfile(model_path):
            self.model = load_model(model_path)
            print('Loaded model from', model_path)
    
    def save(self):
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        model_path = os.path.join(save_dir, self.get_save_file())
        self.model.save(model_path)

    def _board_states_to_inputs(self, board_states):
        inputs = np.array(board_states)
        inputs = np.expand_dims(inputs, axis=3)

        return inputs

    @abstractmethod
    def get_save_file(self):
        pass

    @abstractmethod
    def get_name(self):
        pass
class NetworkB(Network):
    def __init__(self, id=''):
        self.id = id
        model = Sequential()
        model.add(Conv2D(128, (4,4), input_shape=(6, 7, 1)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(64))
        model.add(Activation('relu'))
        model.add(Dense(1))

        # opt = keras.optimizers.RMSprop(learning_rate=0.0001)
        opt = keras.optimizers.Adam()

        model.compile(loss='mean_squared_error',
                      optimizer=opt,
                      metrics=['accuracy'])

        super().__init__(model)

    def get_save_file(self):
        return 'model_128-4_64_64_B{}.h5'.format(self.id)

    def get_name(self):
        return 'B' + str(self.id)