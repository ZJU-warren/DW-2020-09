from Model import GBDT, LR
from sklearn.preprocessing import OneHotEncoder
import numpy as np


class Hybrid:
    def __init__(self):
        self.gbdt = GBDT.generate_clf()
        self.lr = LR.generate_clf()
        self.enc = OneHotEncoder(categories='auto')

    def fit(self, X, y):
        self.gbdt.fit(X, y)
        print('---------gbdt fitted------------')
        temp_X = self.gbdt.apply(X)[:, :, 0]
        self.enc.fit(temp_X)
        print('---------enc fitted------------')
        temp_X = np.concatenate((temp_X, X), axis=1)
        self.lr.fit(temp_X, y)
        print('---------lr fitted------------')

    def predict_proba(self, X):
        temp_X = self.gbdt.apply(X)[:, :, 0]
        temp_X = np.concatenate((temp_X, X), axis=1)
        return self.lr.predict_proba(temp_X)

    def predict(self, X):
        temp_X = self.gbdt.apply(X)[:, :, 0]
        temp_X = np.concatenate((temp_X, X), axis=1)
        return self.lr.predict_proba(temp_X)[:, 1] > 0.5


def generate_clf():
    return Hybrid()