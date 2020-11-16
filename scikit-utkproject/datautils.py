from dataset import *
import sklearn
import os

def parse_date_code(raw_date):
    year = raw_date[0:4]
    month = raw_date[4:6]
    day = raw_date[6:8]
    hour = raw_date[8:10]
    minute = raw_date[10:12]
    second = raw_date[12:14]
    FFF = raw_date[14:] #idk what this is
    return [[year, month, day], [hour, minute, second]]

def parse_image_name(title):
    title_sections = title.split("_")
    age = title_sections[0]
    gender = title_sections[1] #0 is male, 1 is female
    ethnicity = title_sections[2] #0 to 4 = White, Black, Asian, Indian, and Others (like Hispanic, Latino, Middle Eastern).

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
