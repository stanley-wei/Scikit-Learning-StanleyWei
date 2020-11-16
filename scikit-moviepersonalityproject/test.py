from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
import math
# import sklearn

import dataset_utils
import data_utils

def split_extraversion(df):
    df['extraversionnew'] = df.loc[:, 'extraversion']
    column_index = data_utils.get_column_index(df, 'extraversionnew')
    for i in range(len(df)):
        extraversion_val = df.iloc[i, column_index]
        if extraversion_val < (7/3):
            df.iloc[i, column_index] = -1
        elif extraversion_val > (14/3):
            df.iloc[i, column_index] = 1
        else:
            df.iloc[i, column_index] = 0
    return df


def main():
    personality_df = dataset_utils.read_personality_data()

    # personality_df = data_utils.exponential_transform_column(personality_df, "emotional_stability", 2)
    # personality_df = data_utils.log_transform_column(personality_df, "openness", 10)
    # personality_df = data_utils.log_transform_column(personality_df, "conscientiousness", 10)
    # personality_df = data_utils.log_transform_column(personality_df, "agreeableness", 10)

    personality_train, personality_test = data_utils.split_df(personality_df)
    personality_train = data_utils.split_ternary(split_extraversion(personality_train), 'extraversionnew')
    personality_test = data_utils.split_ternary(split_extraversion(personality_test), 'extraversionnew')

    # personality_train = data_utils.log_transform_column(personality_train, "emotional_stability")
    personality_train = data_utils.exponential_transform_column(personality_train, "emotional_stability", 2)
    personality_train = data_utils.log_transform_column(personality_train, "openness", 1.35)
    personality_train = data_utils.exponential_transform_column_to_mean(personality_train, "conscientiousness", 4)
    personality_train = data_utils.exponential_transform_column_to_mean(personality_train, "agreeableness", 3)


    itera = 10000
    weight = "balanced"
    intercept = True
    class_multi = 'auto'
    set_c = float(0.5)
    start = True

    personality_traits = ['openness', 'agreeableness', 'emotional_stability', 'conscientiousness']

    introversion_model = LogisticRegression(max_iter=itera, class_weight=weight, warm_start=start, fit_intercept=intercept, C=set_c, multi_class=class_multi)
    introversion_model.fit(personality_train.loc[:, personality_traits], personality_train.loc[:, 'extraversionnew-'])

    extraversion_model = LogisticRegression(max_iter=itera, class_weight=weight, warm_start=start, fit_intercept=intercept, C=set_c, multi_class=class_multi)
    extraversion_model.fit(personality_train.loc[:, personality_traits], personality_train.loc[:, 'extraversionnew+'])

    introversion_predictions = introversion_model.predict(personality_test.loc[:, personality_traits])
    extraversion_predictions = extraversion_model.predict(personality_test.loc[:, personality_traits])

    extravert_results = [0, 0, 0]
    introvert_results = [0, 0, 0]
    none_results = [0, 0, 0]

    column_index = data_utils.get_column_index(personality_test, 'extraversionnew')
    for i in range(len(personality_test)):
        introversion_prediction = introversion_predictions[i]
        extraversion_prediction = extraversion_predictions[i]
        if extraversion_prediction > 0.99 and extraversion_prediction > 4 * introversion_prediction:
            prediction = 1
            pos = 2
        elif introversion_prediction > 0.99 and introversion_prediction > 4 * extraversion_prediction:
            prediction = -1
            pos = 0
        else:
            prediction = 0
            pos = 1

        true_value = personality_test.iloc[i, column_index]

        if true_value == 1:
            extravert_results[pos] += 1
        elif true_value == -1:
            introvert_results[pos] += 1
        else:
            none_results[pos] += 1

    num_correct = extravert_results[2] + introvert_results[0] + none_results[1]
    accuracy = float(num_correct/len(personality_test))
    num_correct_outlier = extravert_results[2] + introvert_results[0]
    accuracy_outlier = float(num_correct_outlier/(sum(introvert_results) + sum(extravert_results)))
    print("Accuracy: " + str(int(accuracy*100)) + "%")
    print("Edge case accuracy: " + str(int(accuracy_outlier*100)) + "%")
    print("Extravert results: " + str(extravert_results))
    print("None results: " + str(none_results))
    print("Introvert results: " + str(introvert_results))

main()
