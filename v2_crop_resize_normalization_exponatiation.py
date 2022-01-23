import cv2, pydicom, nrrd, os, timeit
import numpy as np
from PIL import Image
from numpy import asarray, save


def path(directory):  # return dcm_list, nrrd_list
    dcm_list, nrrd_list = [], []
    IDs = os.listdir(directory)
    for ID in IDs:
        files = os.listdir(directory + ID)
        for file in files:
            f_name, file_ext = os.path.splitext(file)
            if file_ext == '.nrrd':
                path_nrrd = directory + ID + '/' + file
                name_dcm = f_name[5:] + '.dcm'
                path_dcm = directory + ID + '/' + name_dcm
                nrrd_list.append(path_nrrd)
                dcm_list.append(path_dcm)
    return dcm_list, nrrd_list


def segmentation(path_dcm, path_nrrd):  # return ROI, mask_array
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
    mask_array, header = nrrd.read(path_nrrd)
    mask_array = np.reshape(mask_array, (800, 600))
    mask_array = np.transpose(mask_array)

    # apply nrrd to dcm in array type
    ROI = np.multiply(mask_array, array_2D)

    return ROI, mask_array


def find_contour(array):  # return contour
    contour = []
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            loc = []
            if array[i][j] > 0:
                loc.append(i)
                loc.append(j)
                contour.append(loc)
                break
        for k in range(array.shape[1]):
            loc = []
            if (array[i][array.shape[1] - k - 1] > 0) and (j != (array.shape[1] - k - 1)):
                loc.append(i)
                loc.append(array.shape[1]-k - 1)
                contour.append(loc)
                break
    return contour


def overview(area):  # return mean_value_pixel
    num_pixel = 0
    total_value_pixel = 0
    for i in range(area.shape[0]):
        for j in range(area.shape[1]):
            if area[i][j] > 0:
                num_pixel += 1
                total_value_pixel += area[i][j]
    mean_value_pixel = total_value_pixel/num_pixel
    return mean_value_pixel


def delete_markers(ROI):  # return roi without markers
    mean = overview(ROI)
    for i in range(600):
        for j in range(800):
            if ROI[i][j] >= 230:
                ROI[i][j] = mean
    return ROI


def cut_border(contour, roi):  # return mask_array with border cut
    upper = contour[0][0]
    lower = contour[-1][0]
    for con in contour:
        for i in range(upper, lower+1):
            for j in range(roi.shape[1]):
                temp = (i - con[0])**2 + (j - con[1])**2
                if temp > 0 and temp <= 900:  # size of eraser
                    roi[i][j] = 0
    return roi


def medulla(contour, roi):  # return medulla after resize
    for con in contour:
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                temp = (i - con[0])**2 + (j - con[1])**2
                if temp > 0 and temp <= 6400:  # size of eraser
                    roi[i][j] = 0
    return roi


def exp_value(array_2D):  # return exponential array
    exp = np.multiply(array_2D, array_2D)
    return exp


def crop(roi, contour):  # return rectangle contains only roi area
    upper = contour[0][0]
    lower = contour[-1][0]
    left = contour[0][1]
    right = left
    for i in range(len(contour)):
        if contour[i][1] > right:
            right = contour[i][1]
        if contour[i][1] < left:
            left = contour[i][1]
    roi = roi[upper:lower, left:right]
    return roi


def resize(roi):  # return roi resized to (350, 600)
    if (roi.shape[0] / roi.shape[1]) < (7/12):
        dims = (600, (roi.shape[0] * 600) // roi.shape[1])
    else:
        dims = ((350 * roi.shape[1]) // roi.shape[0], 350)
    roi = cv2.resize(roi, dsize=dims, interpolation=cv2.INTER_NEAREST)
    return roi


root = 'D:/renalUS/U55/'

a, b = path(root)
for i in range(1517, len (a)):
    start = timeit.default_timer()
    roi, mask = segmentation(a[i], b[i])
    print ('step %s.1' %i)
    roi_no_marker = delete_markers(roi)
    print ('step %s.2' %i)
    contour_roi_no_marker = find_contour(roi_no_marker)
    print ('step %s.3' %i)
    roi_no_border = cut_border(contour_roi_no_marker, roi_no_marker)
    print ('step %s.4' %i)
    contour_roi_no_border = find_contour(roi_no_border)
    print ('step %s.5' %i)
    roi_crop = crop(roi_no_border, contour_roi_no_border)
    print ('step %s.6' %i)
    roi_resize = resize(roi_crop)
    print ('step %s.7' %i)    
    contour_roi_resize = find_contour(roi_resize)
    print ('step %s.8' %i)
    medulla_roi_resize = medulla(contour_roi_resize, roi_resize)
    print ('step %s.9' %i)    
    mean_medulla = overview (medulla_roi_resize)
    print ('step %s.10' %i)    
    roi_nor = roi_resize/ mean_medulla # normalized roi
    print ('step %s.11' %i)
    roi_exp = np.multiply(roi_nor, roi_nor)
    print ('step %s.12' %i)    
    save ('D:/renalUS/np_U55/%s.npy' %i, roi_exp)
    stop = timeit.default_timer()
    print ('FINAL STEPPPPP of %s' %i, 'with time:', stop-start)

