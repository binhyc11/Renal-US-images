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


def find_contour(array):  # return contour of an array
    '''
    Input: a array of segmentation with border
    Return: a list contains coordinates of the contour
    '''
    contour = []
    for i in range(array.shape[0]):  # run from left to right
        for j in range(array.shape[1]):
            loc = []
            if array[i][j] > 0:
                loc.append(i)
                loc.append(j)
                contour.append(loc)
                break
        for k in range(array.shape[1]):  # run from right to left
            loc = []
            if (array[i][array.shape[1] - 1 - k] > 0) and (j != (array.shape[1] - 1 - k)):
                loc.append(i)
                loc.append(array.shape[1] - 1 - k)
                contour.append(loc)
                break

    for l in range(array.shape[1]):  # run from top to bottom
        for m in range(array.shape[0]):
            loc = []
            if array[:, l][m] > 0:
                loc.append(m)
                loc.append(l)
                if loc not in contour:
                    contour.append(loc)
                break
            
        for n in range(array.shape[0]):
            loc = []
            if array[:, l][array.shape[0] - 1 - n] > 0 and (m != (array.shape[0] - 1 - n)):
                loc.append(array.shape[0] - 1 - n)
                loc.append(l)
                if loc not in contour:
                    contour.append(loc)
                break
    return contour


def crop(roi, contour):  # return rectangle contains only roi area
    upper = contour[0][0]
    lower = upper
    left = contour[0][1]
    right = left
    roi2 = copy.deepcopy(roi)
    for i in range(len(contour)):
        if contour[i][1] > right:
            right = contour[i][1]
        if contour[i][1] < left:
            left = contour[i][1]
        if contour[i][0] > lower:
            lower = contour[i][0]
        if contour[i][0] < upper:
            upper = contour[i][0]

    start = max(0, upper - 1)
    end = max(0, left - 1)

    roi2 = roi2[start: (lower+1), end: (right+1)]
    return roi2


def get_size (ID, L_flag, R_flag):
    for i in rows:
        if ID == i[0]:
            if L_flag == True:
                length= float(i[1])
            if R_flag == True:
                length= float(i[2])
    return length


def overview(area):  # return mean_value_pixel
    num_pixel = 0
    for i in range(area.shape[0]):
        for j in range(area.shape[1]):
            if area[i][j] > 0:
                num_pixel += 1
    return num_pixel


with open ('C:/Users/binhy/Desktop/patients_list.csv') as file:
    csvreader = csv.reader (file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        if row[1] != 'L NA':
            rows.append (row)
ID_area = []
for i in rows:
    for j in os.listdir('D:/v4_preprocessing/'):
        if 'npy' in j and i[0] == j[:len(i[0])]:
            if 'L' in j:
                length_cm = float (i[1])
            if 'R' in j:
                length_cm = float (i[2])
            img = np.load ('D:/v4_preprocessing/' + j )
            contour = find_contour(img)
            length_pixel = (crop(img, contour).shape)[1]
            area_pixel = overview(img)
            area_cm = area_pixel * length_cm * length_cm  / length_pixel / length_pixel
            ID_area.append ([j[:-4], area_cm])
            
            
i = 0
ID_area_2 = []
while i < 2975:
    temp = ID_area[i][0].split('_')
    if len(temp) == 3:
        ID = temp[0] + '_' + temp[1]
    if len(temp) == 4:
        ID = temp[0] + '_' + temp[1] + '_' + temp[2]
    counter = i
    sum_area = 0
    print ('g', i)
    while counter < 2975:
        if counter == 2974 or (ID not in ID_area[counter+1][0]):
            # something
            sum_area += ID_area[counter][1]
            area_mean = sum_area/ (counter-i+1)
            ID_area_2.append ([ID, area_mean])
            break
        sum_area += ID_area[counter][1]
        counter += 1
    i=counter+1
