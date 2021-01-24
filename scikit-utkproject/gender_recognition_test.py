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

def main():
    from_root = "~/Documents/School/ComputerScience/ahcompsci/Scikit-Learning-StanleyWei/scikit-utkproject/dataset/fiftytwo"
    path = "dataset/fourty/"
    dirs = os.listdir(path)

    train_images, test_images = split_dataset(dirs)

    train_df = add_images_from_dirs(train_images, path)
    test_df = add_images_from_dirs(test_images, path)

    # train_df = train_df.loc[train_df['ethnicity'] == "0"]
    # test_df = test_df.loc[test_df['ethnicity'] == "0"]
    train_x = flatten_image_df(train_df)
    test_x = flatten_image_df(test_df)

    clf = LogisticRegression(random_state = 0, max_iter = 1000) #requires 750 < x < 1000 iterations
    # train_x = np.array(train_df.loc[:, "image"])
    # x_train = train_x.flatten().reshape(len(train_df), -1)
    clf.fit(train_x, train_df.loc[:, "gender"].to_numpy())

    true_male = [0, 0]
    true_female = [0, 0]

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
    # print(coefficients)
    coefficients_array = np.array(coefficients).reshape(len(train_df.image[0]), -1)
    # print(coefficients_array)
    # heatmap = plt.imshow(coefficients_array, cmap = "hot", interpolation = "nearest")
    coefficients_abs = coefficients
    for i in range(len(coefficients_abs)):
        coefficients_abs[i] = abs(coefficients_abs[i])
    coefficients_array_abs = np.array(coefficients_abs).reshape(len(train_df.image[0]), -1)
    heatmap = plt.imshow(coefficients_array_abs, cmap = "hot", vmax = 0.005, interpolation = "nearest")
    # heatmap_extremes = plt.imshow(coefficients_array_abs, vmax = 0.025, cmap = "hot", interpolation = "nearest")
    plt.colorbar(heatmap)
    # plt.colorbar(heatmap_extremes)
    plt.show()

    f = open("gendercoefsfourty.txt", "w")
    f_string = ""
    for i in range(len(clf.coef_)):
        f_string += str(clf.coef_[i]) + ","
    f.write(f_string)
    f.close()

main()
