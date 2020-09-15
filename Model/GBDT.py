from sklearn.ensemble import GradientBoostingClassifier


def generate_clf(learning_rate=0.15,
                 n_estimators=300,
                 max_depth=9,
                 subsample=0.65,
                 max_features='sqrt',
                 verbose=1):
    return GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth,
                                      subsample=subsample, max_features=max_features, verbose=verbose)
