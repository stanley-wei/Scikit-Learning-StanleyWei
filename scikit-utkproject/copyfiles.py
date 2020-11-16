import shutil
import argparse

def make_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("age")
    ap.add_argument("race")
    ap.add_argument("gender")
    args = vars(ap.parse_args())
    return args

def main():
    print(make_args())

main()
