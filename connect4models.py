import tensorflow as tf
from tensorflow import keras
from keras import Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D 


initializer = tf.keras.initializers.RandomNormal(mean=0, stddev=0.01)


def create_model () :
    model = Sequential()
    model.add(Conv2D(48, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(6,7,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(48, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(7, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = 'accuracy')
    return model


def create_model2 () :
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(4, 4), activation='relu', padding='same', input_shape=(6,7,3)))
    model.add(BatchNormalization())
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(BatchNormalization())
    #model.add(Dropout(rate=0.5))
    model.add(Dense(7, activation='softmax'))
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    model.compile(loss = 'categorical_crossentropy', optimizer = opt, metrics = 'accuracy')
    return model



class connectzero(Model):
    def __init__(self):
        super(connectzero, self).__init__()

        self.conv1 = Conv2D(32, 3, strides=2, activation="relu", padding='same')
        self.max1  = MaxPooling2D(3)
        self.bn1   = BatchNormalization()

        self.conv2 = Conv2D(64, 3, activation="relu", padding='same')
        self.bn2   = BatchNormalization()
        
        self.fl = Flatten()
        self.d1 = Dense(512, activation='relu')
        self.bn3 = BatchNormalization()

        self.dp = Dense(128, activation='relu')
        self.dv = Dense(64, activation='relu')

        self.policy = Dense(7, activation='softmax')
        self.value = Dense(1, activation = 'tanh')




        opt = tf.keras.optimizers.Adam(learning_rate=0.1)
        self.compile(loss=['categorical_crossentropy','mean_squared_error'], optimizer=opt)

    def call(self, x):
        x = self.conv1(x)
        x = self.max1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.fl(x)
        x = self.d1(x)
        x = self.bn3(x)
        p = self.dp(x)
        v = self.dv(x)
        policy = self.policy(p)
        value = self.value(v)
        return policy, value
