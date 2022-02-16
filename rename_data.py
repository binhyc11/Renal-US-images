import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

root = 'D:/Pre-processing/data_roi_medulla_wo_exp_&_std/'
ID = 'D:/data/test_wo_exp/5/'


for i in os.listdir(root):
    for j in os.listdir(ID):
        if j[2:-4] == i [4:-4]:
            old = root + i
            new = root + 'roi_' + j [:-4] + '.npy'
            os.rename(old, new)
