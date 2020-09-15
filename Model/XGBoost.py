import xgboost as xgb


def generate_clf(learning_rate=0.15,
                 n_estimators=100,
                 max_depth=3,
                 subsample=0.7,
                 verbose=2):

    param = {'num_class': 2, 'objective': 'multi:softmax', 'learning_rate': learning_rate,
             'max_depth': max_depth, 'subsample': subsample, 'n_estimators': n_estimators,
             'nthread': -1, 'verbosity': verbose}

    return xgb.XGBClassifier(**param)
