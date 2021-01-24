from sklearn import svm
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import SGDClassifier, LogisticRegression, Lars
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

from datautils import *
# from dataset.data_keys import *
import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns

def main():
    csv_path = "dataset/fer2013/fer2013p4.csv"
    fer_df = pd.read_csv(csv_path)
    print(fer_df)

    image_train, image_test, emotion_train, emotion_test = train_test_split(fer_df.loc[:, "pixels"], fer_df.loc[:, "emotion"], test_size = 0.2, random_state = 42)

    model = Lars

main()
