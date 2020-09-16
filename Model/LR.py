from sklearn.linear_model import LogisticRegression


def generate_clf():
    return LogisticRegression(penalty='l2', verbose=True, n_jobs=-1, solver='sag', max_iter=100)
