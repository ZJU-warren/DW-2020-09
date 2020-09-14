import time
import joblib


class ModelProxy:
    clf = None

    def __init__(self, clf=None, data_link=None):
        if data_link is None:
            self.clf = clf
        else:
            self.load(data_link)

    # fit (X, y)
    def fit(self, x, y):
        t1 = time.time()
        self.clf.fit(x, y)
        t2 = time.time()
        print('train time used %d s' % (t2 - t1))

    # save model
    def save(self, store_link):
        joblib.dump(self.clf, store_link)

    # load model
    def load(self, store_link):
        self.clf = joblib.load(store_link)

    # predict
    def predict(self, x):
        y = self.clf.predict(x)
        return y

    # predict_proba
    def predict_proba(self, x):
        y = self.clf.predict_proba(x)
        return y