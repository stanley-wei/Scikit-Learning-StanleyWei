import numpy as np
import pandas as pd
import sklearn
import math

def split_df(df):
    train_max = int(0.8 * len(df))
    # test_max = int(0.9 * len(df))

    split_df = np.split(df, [train_max])

    df_train = pd.DataFrame(split_df[0])
    df_test = pd.DataFrame(split_df[1])
    # df_test = pd.DataFrame(split_df[2])

    return df_train, df_test

def get_column_index(df, column_name):
    df_columns = df.columns
    for i in range(len(df_columns)):
        column_found = df_columns[i]
        if column_found == column_name:
            return i
    return -1

def log_transform_column(df, column_name, base):
    column_index = get_column_index(df, column_name)
    for i in range(len(df)):
        df.iloc[i, column_index] = math.log(df.iloc[i, column_index], base)
    return df

def exponential_transform_column(df, column_name, power):
    column_index = get_column_index(df, column_name)
    for i in range(len(df)):
        # df.iloc[i, column_index] = base**(df.iloc[i, column_index]) * df.iloc[i, column_index]
        df.iloc[i, column_index] = df.iloc[i,column_index]**power
    return df

def exponential_transform_column_to_mean(df, column_name, power):
    column_index = get_column_index(df, column_name)
    column_mean = float(df.loc[:, column_name].mean())
    for i in range(len(df)):
        # df.iloc[i, column_index] = base**(df.iloc[i, column_index]) * df.iloc[i, column_index]
        adjusted_to_mean = df.iloc[i, column_index] - column_mean
        df.iloc[i, column_index] = (abs(adjusted_to_mean)**power) * (abs(adjusted_to_mean)/adjusted_to_mean)
    return df

def split_ternary(df, column_name):
    df[column_name + "-"] = df.loc[:, column_name]
    df[column_name + "+"] = df.loc[:, column_name]
    column_index = get_column_index(df, column_name)

    to_categorize = [column_name + "-", column_name + "+"]
    for column in to_categorize:
        column_index = get_column_index(df, column)
        if column[-1] == "-":
            objective = -1
        else:
            objective = 1
        for i in range(len(df)):
            column_val = df.iloc[i, column_index]
            if column_val == objective:
                df.iloc[i, column_index] = 1
            else:
                df.iloc[i, column_index] = 0

    return df
