from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from datautils import *
from dataset.data_keys import *
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

def split_dataset(dataset):
    eighty_percent = int(0.8 * len(dataset))

    dataset_train = dataset[0:eighty_percent]
    dataset_test = dataset[eighty_percent:]

    return dataset_train, dataset_test

def equalize_binary_variable(key, df, path):
    zeroes = df.loc[df[key] == "0"]
    ones = df.loc[df[key] == "1"]

    num_zeroes = len(zeroes)
    num_ones = len(ones)

    if num_zeroes > num_ones:
        to_find = num_ones
    else:
        to_find = num_zeroes

    new_df = pd.DataFrame({'age': [], 'gender': [], 'ethnicity': [], 'date': [], 'image': []})
    for image_name in df:
        if image_name[-3:] == "jpg":
            age, gender, ethnicity, date = parse_image_name(image_name)
            if gender == 0 and num_zeroes < to_find:
                image = cv2.imread(path + image_name)
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                new_row = {'age': age, 'gender': gender, 'ethnicity': ethnicity, 'date': date, 'image': grayscale_image}
                new_df = new_df.append(new_row, ignore_index = True)
                num_zeroes += 1
            elif gender == 1 and num_ones < to_find:
                image = cv2.imread(path + image_name)
                grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                new_row = {'age': age, 'gender': gender, 'ethnicity': ethnicity, 'date': date, 'image': grayscale_image}
                new_df = new_df.append(new_row, ignore_index = True)
                num_ones += 1
            elif num_ones >= to_find and num_zeroes >= to_find:
                break
    return new_df

def main():
    from_root = "~/Documents/School/ComputerScience/ahcompsci/Scikit-Learning-StanleyWei/scikit-utkproject/dataset/fiftytwo"
    path = "dataset/fourty/"
    dirs = os.listdir(path)

    train_images, test_images = split_dataset(dirs)

    train_df = pd.DataFrame({'age': [], 'gender': [], 'ethnicity': [], 'date': [], 'image': []})
    # train_df = equalize_binary_variable("gender", train_df, path)
    for image_name in train_images:
        if image_name[-3:] == "jpg":
            age, gender, ethnicity, date = parse_image_name(image_name)
            image = cv2.imread(path + image_name)
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            new_row = {'age': age, 'gender': gender, 'ethnicity': ethnicity, 'date': date, 'image': grayscale_image}
            train_df = train_df.append(new_row, ignore_index = True)

    test_df = pd.DataFrame({'age': [], 'gender': [], 'ethnicity': [], 'date': [], 'image': []})
    for image_name in test_images:
        if image_name[-3:] == "jpg":
            age, gender, ethnicity, date = parse_image_name(image_name)
            image = cv2.imread(path + image_name)
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            new_row = {'age': age, 'gender': gender, 'ethnicity': ethnicity, 'date': date, 'image': grayscale_image}
            test_df = test_df.append(new_row, ignore_index = True)

    # train_df = train_df.loc[train_df['ethnicity'] == "0"]
    # test_df = test_df.loc[test_df['ethnicity'] == "0"]

    clf = LogisticRegression(random_state = 0, max_iter = 1000) #requires 750 < x < 1000 iterations
    # train_x = np.array(train_df.loc[:, "image"])
    # x_train = train_x.flatten().reshape(len(train_df), -1)
    train_image_set = train_df.loc[:, "image"]
    train_x = np.array([train_image_set.iloc[i].flatten() for i in range(0, len(train_df))])
    clf.fit(train_x, train_df.loc[:, "gender"].to_numpy())

    true_male = [0, 0]
    true_female = [0, 0]

    test_image_set = test_df.loc[:, "image"]
    test_x = np.array([test_image_set.iloc[i].flatten() for i in range(0, len(test_df))])

    for i in range(len(test_df)):
        true_value = int(test_df.iloc[i, 1])
        prediction = int(clf.predict([test_x[i]])[0])
        if true_value == 0:
            if prediction == 0:
                true_male[0] += 1
            else:
                true_male[1] += 1
        elif true_value == 1:
            if prediction == 1:
                true_female[0] += 1
            else:
                true_female[1] += 1
        # print(prediction)

    print("Male results: " + str(true_male))
    print("Female results: " + str(true_female))

    male_accuracy = float(true_male[0] / (true_male[0] + true_male[1]))
    female_accuracy = float(true_female[0] / (true_female[0] + true_female[1]))
    overall_accuracy = float((true_male[0] + true_female[0]) / (true_male[0] + true_male[1] + true_female[0] + true_female[1]))

    print("Male accuracy: " + str(male_accuracy))
    print("Female accuracy: " + str(female_accuracy))
    print("Overall accuracy: " + str(overall_accuracy))

    coefficients = clf.coef_
    coefficients_array = np.array(coefficients).reshape(len(grayscale_image), -1)
    # heatmap = plt.imshow(coefficients_array, cmap = "gist_gray", interpolation = "nearest")
    coefficients_abs = coefficients
    for i in range(len(coefficients_abs)):
        coefficients_abs[i] = abs(coefficients_abs[i])
    coefficients_array_abs = np.array(coefficients_abs).reshape(len(grayscale_image), -1)
    heatmap_extremes = plt.imshow(coefficients_array_abs, vmax = 0.025, cmap = "hot", interpolation = "nearest")
    # plt.colorbar(heatmap)
    plt.colorbar(heatmap_extremes)
    plt.show()

main()
