from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import sys;sys.path.append('../')
from Model import GBDT
from Model import XGBoost
from Model import LightGBM
from Model import Hybrid_LR_GBDT
from Model import DNN
import pandas as pd
from Model.ModelProxy import ModelProxy
from sklearn.metrics import f1_score, roc_curve, auc
import DataLinkSet as DLSet
import datetime
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


def get_clf(choice):
    if choice == 'GBDT':
        return GBDT.generate_clf()
    elif choice == 'XGBoost':
        return XGBoost.generate_clf()
    elif choice == 'LightGBM':
        return LightGBM.generate_clf()
    elif choice == 'Hybrid_LR_GBDT':
        return Hybrid_LR_GBDT.generate_clf()
    elif choice == 'DNN':
        return DNN.generate_clf()


def main():
    df_clean_train = pd.read_csv(DLSet.clean_train_link)
    df_clean_test = pd.read_csv(DLSet.clean_test_link)

    # confirm features
    features_used = df_clean_train.columns.difference(['id', 'earliesCreditLine', 'isDefault', 'n2.2', 'n2.3'])

    # # # state clf
    # # model_choice = 'GBDT'
    model_choice = 'Hybrid_LR_GBDT'
    # # model_choice = 'DNN'
    # # model_choice = 'XGBoost'
    # # model_choice = 'LightGBM'
    model = ModelProxy(clf=get_clf(model_choice))
    # # pca = PCA(n_components=30)
    # #

    # training
    train_X = df_clean_train[features_used].values

    # scaler = StandardScaler().fit(train_X)
    # train_X = scaler.transform(train_X)
    # train_X = pca.fit_transform(train_X)
    train_y = df_clean_train['isDefault'].values

    ye1 = np.where(train_y == 1.0)
    X_copy = train_X[ye1].copy()
    y_copy = train_y[ye1].copy()

    print(train_X.shape, X_copy.shape)
    train_X = np.concatenate((train_X, X_copy))
    train_y = np.concatenate((train_y, y_copy))
    #
    #
    #
    # # # add PCA
    model.fit(train_X, train_y)
    model.save(DLSet.model_link % (model_choice, datetime.datetime.today()))

    # evaluate model
    model = ModelProxy(data_link=DLSet.model_link % ('Hybrid_LR_GBDT', '2020-09-16 12:56:55.730132'))
    pred_y = model.predict(train_X)
    result = f1_score(train_y, pred_y)
    print('total F1 = {}'.format(result))

    if model_choice in ['DNN']:
        pred_y_prob = model.predict_proba(train_X)
    else:
        pred_y_prob = model.predict_proba(train_X)[:, 1]

    fpr, tpr, thresholds = roc_curve(train_y, pred_y_prob)
    print('auc is', auc(fpr, tpr))

    # predicting
    test_X = df_clean_test[features_used].values
    # test_X = scaler.transform(test_X)
    # test_X = pca.transform(test_X)

    if model_choice in ['DNN']:
        pred_y_prob = model.predict_proba(test_X)
    else:
        pred_y_prob = model.predict_proba(test_X)[:, 1]

    id_list = df_clean_test['id'].values
    num = len(id_list)
    with open(DLSet.result_link % datetime.datetime.today(), 'w') as f:
        f.write('id,isDefault\n')
        for i in range(num):
            f.write('%d,%f\n' % (id_list[i], pred_y_prob[i]))


if __name__ == '__main__':
    main()
