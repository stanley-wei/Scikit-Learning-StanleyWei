from dataset import *
import sklearn
import os
import pandas as pd
import numpy as np
import cv2

def parse_date_code(raw_date):
    year = int(raw_date[0:4])
    month = int(raw_date[4:6])
    day = int(raw_date[6:8])
    hour = int(raw_date[8:10])
    minute = int(raw_date[10:12])
    second = int(raw_date[12:14])
    FFF = int(raw_date[14:]) #idk what this is
    return [[year, month, day], [hour, minute, second]]

def reorder_ethnicity_col(df):
    ethnicity_col = df.ethnicity
    for i in range(len(ethnicity_col)):
        if ethnicity_col[i] == 1:
            ethnicity_col[i] = 4
        elif ethnicity_col[i] == 2:
            ethnicity_col[i] = 1
        elif ethnicity_col[i] == 4:
            ethnicity_col[i] = 2
    df.ethnicity = ethnicity_col
    return df

def add_images_from_dirs(image_paths, path):
    df = pd.DataFrame({'age': [], 'gender': [], 'ethnicity': [], 'date': [], 'image': []})
    # train_df = equalize_binary_variable("gender", train_df, path)
    for image_name in image_paths:
        if image_name[-3:] == "jpg":
            age, gender, ethnicity, date = parse_image_name(image_name)
            image = cv2.imread(path + image_name)
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            new_row = {'age': age, 'gender': gender, 'ethnicity': ethnicity, 'date': date, 'image': grayscale_image}
            df = df.append(new_row, ignore_index = True)
    return df

def flatten_image_df(df):
    image_column = df.loc[:, "image"]
    image_df = np.array([image_column.iloc[i].flatten() for i in range(0, len(image_column))])
    return image_df

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

def parse_image_name(title):
    title_sections = title.split("_")
    age = int(title_sections[0])
    gender = int(title_sections[1]) #0 is male, 1 is female
    ethnicity = int(title_sections[2]) #0 to 4 = White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).

    raw_date = title_sections[3].split(".")[0]
    date = parse_date_code(raw_date)

    return age, gender, ethnicity, date

# def main():
#     from_root = "~/Documents/School/ComputerScience/ahcompsci/Scikit-Learning-StanleyWei/include/datasets/utkcropped"
#     path = "include/datasets/utkcropped"
#     dirs = os.listdir(path)
#     for i in range(20):
#         print(parse_image_name(dirs[i]))
#
# main()
