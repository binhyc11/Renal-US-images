import cv2, pydicom, nrrd, os
import numpy as np
from PIL import Image
from numpy import asarray, save

def path(directory):
    dcm_list, nrrd_list = [], []
    IDs = os.listdir(directory)
    for ID in IDs:
        files = os.listdir(directory + ID)
        for file in files:
            f_name, file_ext = os.path.splitext (file)
            if file_ext == '.nrrd':
                path_nrrd = directory + ID + '/' + file
                name_dcm = f_name [5:] +'.dcm'
                path_dcm = directory + ID + '/' + name_dcm
                nrrd_list.append (path_nrrd)
                dcm_list.append (path_dcm)
    return dcm_list, nrrd_list


def segmentation(path_dcm, path_nrrd):
    # convert dcm --> array --> RGB_jpg --> gray_jpg --> array
    dcm_to_array = pydicom.dcmread(path_dcm)
    array = dcm_to_array.pixel_array.astype(float)
    array = (np.maximum(array, 0)/array.max()) * 255.0
    array = np.uint8(array)
    jpg = Image.fromarray(array)
    jpg.save('D:/l-0.jpg')
    jpg = cv2.imread ('D:/l-0.jpg')
    gray_jpg = cv2.cvtColor(jpg, cv2.COLOR_BGR2GRAY)
    array_2D = asarray (gray_jpg)
    
    # nrrd --> array  --> transposing
    mask_array, header = nrrd.read(path_nrrd)
    mask_array = np.reshape (mask_array, (800, 600))
    mask_array = np.transpose(mask_array)
    
    # apply nrrd to dcm in array type
    ROI = np.multiply(mask_array, array_2D)
    
    # find contour of mask
    contour = []
    for i in range (600):
        for j in range (800):
            loc = []
            if mask_array[i][j] > 0:
                loc.append(i)
                loc.append(j)
                contour.append(loc)
                break
        for k in range (800):
            loc = []
            if (mask_array[i][799 - k] > 0) and (j != (799-k)):
                loc.append(i)
                loc.append(799-k)
                contour.append(loc)
                break
    return ROI, contour, mask_array

def overview (ROI):
    num_pixel = 0
    total_value_pixel = 0
    for i in range (600):
        for j in range (800):
            if ROI[i][j] > 0:
                num_pixel += 1
                total_value_pixel += ROI[i][j]
    mean_value_pixel = total_value_pixel/num_pixel
    return mean_value_pixel, num_pixel, total_value_pixel

def delete_markers (ROI):
    mean, _, _ = overview(ROI)
    for i in range (600):
        for j in range (800):
            if ROI[i][j] >= 230:
                ROI[i][j] = mean
    return ROI

def cut_border (contour, mask_array):
    for con in contour:
        for i in range (600):
            for j in range (800):
                temp = (i - con[0])**2 + (j - con[1])**2
                if temp > 0 and temp <= 900:  # size of eraser
                    mask_array[i][j] = 0
    return mask_array

def exp_value (array_2D):
    exp = np.multiply (array_2D, array_2D)
    return exp


root = 'D:/renalUS/B65/'
a, b = path (root)
for i in range (len(a)):
    roi, contour, mask = segmentation (a[i], b[i])
    mask_no_border = cut_border (contour, mask)
    roi_no_border = np.multiply (mask_no_border, roi)
    roi_exp = exp_value (roi_no_border)
    save ('D:/renalUS/np_B65/%s.npy' %i, roi_exp)
    print(i)






