import pandas as pd
import numpy as np
import DataLinkSet as DLSet


# replace str by real number
def func(x, str2num):
    return str2num[x]


def replace_str2num(df_raw, each, columns):
    length = columns.shape[0]
    str2num = dict()
    for i in range(length):
        str2num[columns[i]] = i + 1
    df_raw[each] = df_raw[each].fillna('NULL')
    df_raw[each] = df_raw[each].apply(func, str2num=str2num)
    return df_raw


def replace(df_raw, store_link):
    for each in ['grade', 'subGrade', 'issueDate']:
        columns = df_raw.drop_duplicates(each, keep='last').fillna('NULL')[[each]].values[:, 0]
        columns = np.sort(columns)
        df_raw = replace_str2num(df_raw, each, columns)

    columns = np.array(['NULL', '< 1 year', '1 year', '2 years', '3 years', '4 years',
                        '5 years', '6 years', '7 years', '8 years', '9 years', '10+ years'])
    df_raw = replace_str2num(df_raw, 'employmentLength', columns)
    print('------------------------- stage 1 finish ------------------------- ')

    # replace nan by mode
    df_raw = df_raw.fillna(df_raw.mean()[0])

    # check the columns which contain nan
    isnan_columns = df_raw.isnull().any(axis=0)
    nan_columns = isnan_columns[isnan_columns == True].index.tolist()  # do not rpl == with is
    print(nan_columns)
    print('------------------------- stage 2 finish ------------------------- ')

    # store result
    df_raw.to_csv(store_link, index=False)


def main():
    # load raw data
    # replace(pd.read_csv(DLSet.raw_train_link), DLSet.clean_train_link)
    # replace(pd.read_csv(DLSet.raw_test_link), DLSet.clean_test_link)
    pass


if __name__ == '__main__':
    main()

