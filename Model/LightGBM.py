from lightgbm import LGBMRegressor


def generate_clf():
    return LGBMRegressor(num_leaves=30,
                         max_depth=5,
                         learning_rate=.02,
                         n_estimators=1000,
                         subsample_for_bin=5000,
                         min_child_samples=200,
                         colsample_bytree=.2,
                         reg_alpha=.1,
                         reg_lambda=.1,
                         silent=False)
