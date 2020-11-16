import numpy as np
import pandas as pd
import sklearn

def read_personality_data():
    df = pd.read_csv('2018-personality-data.csv')
    # df = df.dropna()
    return df

def read_ratings_data():
    df = pd.read_csv('2018-ratings.csv')
    df = df.dropna()
    return df

# def main():
#     df = read_personality_data()
#     print(df)
#
# main()
