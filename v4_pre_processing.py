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


def path(directory):  # return dcm_list, nrrd_list
    '''
    Input: directory of .dcm and .nrrd files
    Return: List of .dcm paths and .nrrd path
    '''
    dcm_list, nrrd_list = [], []
    IDs = os.listdir(directory)
    for ID in IDs:
        files = os.listdir(directory + ID)
        for file in files:
            f_name, file_ext = os.path.splitext(file)
            if file_ext == '.nrrd':
                path_nrrd =  ID + '/' + file
                name_dcm = f_name[5:] + '.dcm'
                if name_dcm in files:
                    path_dcm = ID + '/' + name_dcm
                    nrrd_list.append(path_nrrd)
                    dcm_list.append(path_dcm)
    return dcm_list, nrrd_list


def segmentation(path_dcm, path_nrrd):  # return ROI
    '''
    Input: path of .dcm file and according path of .nrrd file
    Return: ROI: segmentation of the kidney with border
            mask_array: a (600, 800) array of .nrrd file
    '''
    # convert dcm --> array --> RGB_jpg --> gray_jpg --> array
    dcm_to_array = pydicom.dcmread(path_dcm)
    array = dcm_to_array.pixel_array.astype(float)
    array = (np.maximum(array, 0)/array.max()) * 255.0
    array = np.uint8(array)
    jpg = Image.fromarray(array)
    jpg.save('D:/l-0.jpg')
    jpg = cv2.imread('D:/l-0.jpg')
    gray_jpg = cv2.cvtColor(jpg, cv2.COLOR_BGR2GRAY)
    array_2D = asarray(gray_jpg)

    # nrrd --> array  --> transposing
    mask_array, _ = nrrd.read(path_nrrd)
    mask_array = np.reshape(
        mask_array, (mask_array.shape[0], mask_array.shape[1]))
    mask_array = np.transpose(mask_array).astype(float)

    # apply nrrd to dcm in array type
    ROI = np.multiply(mask_array, array_2D).astype(float)

    return ROI


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


def optimal_rotation(roi):
    angle = 0
    degree = []

    while angle < 181:
        degree.append(angle)
        angle += 5

    contour_mask = find_contour(roi)
    crop_mask = crop(roi, contour_mask)
    width_min = min(crop_mask.shape[0], crop_mask.shape[1])

    best_angle = 0
    for k in degree:
        ro = rotate(crop_mask, k, resize=True)
        contour_ro = find_contour(ro)
        crop_ro = crop(ro, contour_ro)
        ro_width = min(crop_ro.shape[0], crop_ro.shape[1])

        if ro_width < width_min:
            width_min = ro_width
            best_angle = k

    if best_angle == 0:
        rotated_roi = crop_mask
    else:
        rotated_roi = rotate(crop_mask, best_angle, resize=True)
        contour_ro = find_contour(rotated_roi)
        rotated_roi = crop (rotated_roi, contour_ro)
        
        if rotated_roi.shape[0] > rotated_roi.shape[1]:
            rotated_roi = rotate(crop_mask, best_angle + 90, resize=True)    
            contour_roi = find_contour(rotated_roi)
            rotated_roi = crop (rotated_roi, contour_roi)    
    return rotated_roi

def get_size (ID, L_flag, R_flag):
    for i in rows:
        if ID == i[0]:
            if L_flag == True:
                length= float(i[1])
            if R_flag == True:
                length= float(i[2])
    return length

def resizing(roi, length):  # return roi resized to (140, 140)
    roi2 = copy.deepcopy(roi)

    dims = (round(length * 10), round(roi2.shape[0] * 10 * length/ roi2.shape[1]) )
    
    roi2 = cv2.resize(roi2, dsize=dims, interpolation=cv2.INTER_NEAREST)
    
    shape_diff = np.array((100, 140)) - np.array(roi2.shape)
    new_roi = np.lib.pad(roi2, ((0, shape_diff[0]), (0, shape_diff[1])),
                         'constant', constant_values=(0))
    return new_roi


def exp_value(array_2D):  # return exponential array
    exp = np.multiply(array_2D, array_2D)
    return exp

def scale (roi2): ### scale min of roi = 0, max = 255; background = 0
    roi = copy.deepcopy(roi2)
    ### scale min of roi = 0
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            if roi[i][j] > 0:  
                roi[i][j] -= roi[roi >0].min()
                
    ### scale max of roi = 255
    roi /= roi.max()
    roi *= 255.0
    
    return roi

def border_medulla(roi):  # return roi for no border, medulla for medulla area
    contour = find_contour(roi)
    medulla = copy.deepcopy(roi)

    upper = contour[0][0]
    lower = upper

    for i in contour:
        if i[0] > lower:
            lower = i[0]
        if i[0] < upper:
            upper = i[0]

    width_squared = (lower - upper + 1) ** 2

    for con in contour:
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                temp = (i - con[0])**2 + (j - con[1])**2
                # 10% of width for cutting border
                if temp > 0 and (temp <= (width_squared // 400)):
                    roi[i][j] = 0
                # 35% of width for medulla
                if temp > 0 and (temp <= (width_squared * 49 // 400)):
                    medulla[i][j] = 0
    return roi, medulla


def stadardization(roi, med):
    mean_roi = np.mean(roi[roi > 0])
    mean = np.mean(med[med > 0])
    std = np.std(med[med > 0])
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            if roi[i][j] > 230*230:  # remove markers
                roi[i][j] = mean_roi
    roi2 = (roi - mean)/ std
    return roi2

def remove_outliers (roi, med):
    
    mean = np.mean(med[med > roi.min()])
    std = np.std(med[med > roi.min()])
 
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            if roi[i][j] > (mean + 2*std):  # remove markers
                roi[i][j] = (mean + 2*std)
    return roi
            

root = 'D:/renalUS/data/'
a, b = path(root)
bugs =[]
with open ('C:/Users/binhy/Desktop/patients_list.csv') as file:
    csvreader = csv.reader (file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        if row[1] != 'L NA':
            rows.append (row)
IDs = []
for i in rows:
    IDs.append (i[0])


for i in range(len(a)):
    L_flag, R_flag = False, False
    
    ID = a[i].split('/')[0]
    if ID in IDs:
        if 'l' in a[i].split('/')[1]:
            L_flag = True
        if 'r' in a[i].split('/')[1]:
            R_flag = True
        
        start = timeit.default_timer()
    
        seg = segmentation(root+a[i], root+b[i])
      
        ro = optimal_rotation(seg)

        length = get_size (ID, L_flag, R_flag)
        
        resized = resizing(ro, length)
        
        roi, med = border_medulla(resized)
        
        SD_roi = remove_outliers(roi, med)
        
        exp_roi = exp_value (SD_roi)
        
        scale_roi = scale (exp_roi)

        if i < 50:
            plt.imshow(scale_roi, cmap = 'gray')   #### roi after scale
            plt.axis('off')
            plt.show()
        if np.mean (med) == 0:
            bugs.append (a[i])
            print ('goddamn bug over here')
        if L_flag == True:
            save('D:/v3_preprocessing/' + ID + '_L_' + '%s.npy' % i, scale_roi)
            plt.imshow(scale_roi, cmap = 'gray')   #### roi after scale
            plt.axis('off')                  
            plt.savefig('D:/v3_preprocessing/'+ ID + '_L_' + '%s.png' % i , bbox_inches='tight', pad_inches = 0)
            plt.close()
        if R_flag == True:
            save('D:/v3_preprocessing/' + ID + '_R_' + '%s.npy' % i, scale_roi)
            plt.imshow(scale_roi, cmap = 'gray')   #### roi after scale
            plt.axis('off')                  
            plt.savefig('D:/v3_preprocessing/'+ ID + '_R_' + '%s.png' % i , bbox_inches='tight', pad_inches = 0)
            plt.close()
        stop = timeit.default_timer()
    
        print('FINAL STEPPPPP of %s' % i, 'with time:', stop-start)
