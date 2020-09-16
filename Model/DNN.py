from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
import tensorflow as tf


def generate_clf():
    return DNN()


class DNN:
    def __init__(self):
        self.model = Network()
        self.model.compile(optimizer=tf.keras.optimizers.Adam(),
                           loss='mse',
                           metrics=['mae', 'accuracy', 'binary_accuracy'])

    def fit(self, X, y):
        self.model.fit(X, y, epochs=3, batch_size=50)

    def predict_proba(self, X):
        return self.model.predict(X)[:, 0]

    def predict(self, X):
        pred_y = self.model.predict(X)[:, 0]
        return (pred_y < 0.5).astype(int)


class Network(Model):
    def __init__(self):
        super(Network, self).__init__()
        self.dense_array = []
        self.dense_array.append(Dense(72, activation='relu'))
        self.dense_array.append(Dense(36, activation='relu'))
        self.dense_array.append(Dense(18, activation='relu'))
        self.dense_array.append(Dense(9, activation='softmax'))
        self.dense_array.append(Dense(1, activation='sigmoid'))

    def call(self, x):
        n = len(self.dense_array)
        for i in range(n):
             x = self.dense_array[i](x)
        return x

