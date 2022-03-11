import cv2
import pydicom
import nrrd
import os, csv
import timeit
import copy
import numpy as np
from PIL import Image
from numpy import asarray, save
from skimage.transform import rotate
import matplotlib.pyplot as plt

with open ('D:/v3_preprocessing/list_IDs_splitted_random_0.csv') as file:
    csvreader = csv.reader (file)
    header_IDs = []
    header_IDs = next(csvreader)
    IDs = []
    for row in csvreader:
            IDs.append (row)
        
with open ('D:/v3_preprocessing/files_splitted_random_0.csv') as file:
    csvreader = csv.reader (file)
    header_files = []
    header_files = next(csvreader)
    files = []
    for row in csvreader:
            files.append (row)
            
train_0_ID, train_1_ID, val_0_ID, val_1_ID, test_0_ID, test_1_ID = [],[],[],[],[],[]
train_0_file, train_1_file, val_0_file, val_1_file, test_0_file, test_1_file = [],[],[],[],[],[]

for i in IDs:
    if i[0] != '':
        train_0_ID.append (i[0])
    if i[1] != '':
        train_1_ID.append (i[1])
    if i[2] != '':
        val_0_ID.append (i[2])
    if i[3] != '':
        val_1_ID.append (i[3])
    if i[4] != '':
        test_0_ID.append (i[4])
    if i[5] != '':
        test_1_ID.append (i[5])

for i in files:
    if i[0] != '':
        train_0_file.append (i[0])
    if i[1] != '':
        train_1_file.append (i[1])
    if i[2] != '':
        val_0_file.append (i[2])
    if i[3] != '':
        val_1_file.append (i[3])
    if i[4] != '':
        test_0_file.append (i[4])
    if i[5] != '':
        test_1_file.append (i[5])
        
        
def combination (file_list, ID_list):
    combination = []

    for j in ID_list:
        L_list, R_list = [], []
        for i in file_list:
            if j == i[:len (j)]:
                 
                if 'L' in i:
                    L_list.append (i)
                if 'R' in i:
                    R_list.append (i)

        for k in L_list:
            for l in R_list:
                combination.append ([k, l])
    return combination
    
train_0_com = combination (train_0_file, train_0_ID)
train_1_com = combination (train_1_file, train_1_ID)

val_0_com = combination (val_0_file, val_0_ID)
val_1_com = combination (val_1_file, val_1_ID)

test_0_com = combination (test_0_file, test_0_ID)
test_1_com = combination (test_1_file, test_1_ID)
        