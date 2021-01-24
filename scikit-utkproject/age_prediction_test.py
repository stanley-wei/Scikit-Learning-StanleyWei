from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, Lars
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

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
    path = "dataset/whitemensmall/"
    dirs = os.listdir(path)
    main_df = add_images_from_dirs(dirs, path)

    train_images, test_images = train_test_split(main_df.loc[:, "image"], main_df.loc[:, "gender"])

    # train_df = train_df.loc[train_df['ethnicity'] == "0"]
    # test_df = test_df.loc[test_df['ethnicity'] == "0"]
    train_x = flatten_image_df(train_images)
    test_x = flatten_image_df(test_images)

    clf = Lars()
    # train_x = np.array(train_df.loc[:, "image"])
    # x_train = train_x.flatten().reshape(len(train_df), -1)
    clf.fit(train_x, train_df.loc[:, "age"].to_numpy())

    coefficients = clf.coef_
    # print(coefficients)
    coefficients_array = np.array(coefficients).reshape(len(train_df.image[0]), -1)
    # print(coefficients_array)
    # heatmap = plt.imshow(coefficients_array, cmap = "hot", interpolation = "nearest")
    coefficients_abs = coefficients
    for i in range(len(coefficients_abs)):
        coefficients_abs[i] = abs(coefficients_abs[i])
    coefficients_array_abs = np.array(coefficients_abs).reshape(len(train_df.image[0]), -1)
    heatmap = plt.imshow(coefficients_array_abs, cmap = "hot", interpolation = "nearest")
    # heatmap_extremes = plt.imshow(coefficients_array_abs, vmax = 0.025, cmap = "hot", interpolation = "nearest")
    plt.colorbar(heatmap)
    # plt.colorbar(heatmap_extremes)
    plt.show()

main()
