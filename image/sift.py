import os
import argparse
import cv2 as cv
import numpy as np


def run_sift(args):
    img = cv.imread(os.path.join(args.cache_dir, 'cd2ffdfea31b666fb5e6ef308e34276c.jpg'))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    print('kp = {}'.format(kp))
    print('des = {}'.format(des))
    print('des shape= {}'.format(des.shape))
    print('des[0] = {}'.format(des[0]))
    # print('kp shape = {}'.format(kp.shape))
    # print('des shape = {}'.format(des.shape))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", type=str, default="/cluster/home/hjjiang/PR-project/data/train_images/")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument("--n_neighbors", type=int, default=5)
    args = parser.parse_args()
    run_sift(args)
