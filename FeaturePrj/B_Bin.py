import sys;sys.path.append('../')
import pandas as pd
import DataLinkSet as DLSet


def main():
    df_train = pd.read_csv(DLSet.clean_train_link)
    df_test = pd.read_csv(DLSet.clean_train_link)

    for each in df.columns:
        df_temp = df.drop_duplicates(each, keep='last')
        print(each, df_temp.shape[0])
        if df_temp.shape[0] > 0:
            df[each] = pd.qcut(df[each], 10)



if __name__ == '__main__':
    main()