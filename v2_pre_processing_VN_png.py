import cv2, pydicom, nrrd, os, timeit, copy
import numpy as np
from PIL import Image
from numpy import asarray, save
from skimage.transform import rotate
import matplotlib.pyplot as plt

def segmentation(path_dcm, path_nrrd):  # return ROI, mask_array
    '''
    Input: path of .dcm file and according path of .nrrd file
    Return: ROI: segmentation of the kidney with border
            mask_array: an array of .nrrd file
    '''
    
    # convert dcm --> array --> RGB_jpg --> gray_jpg --> array
    jpg = cv2.imread(path_dcm)
    mask_array = np.asarray(Image.open(path_nrrd))
    gray_jpg = cv2.cvtColor(jpg, cv2.COLOR_BGR2GRAY)
    array_2D = asarray(gray_jpg)

    # apply nrrd to dcm in array type
    ROI = np.multiply(mask_array, array_2D).astype (float)

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
        for k in range(array.shape[1]): # run from right to left
            loc = []
            if (array[i][array.shape[1] - 1 - k] > 0) and (j != (array.shape[1] - 1 - k)):
                loc.append(i)
                loc.append(array.shape[1] - 1 - k)
                contour.append(loc)
                break
            
    for l in range (array.shape[1]): # run from top to bottom
        for m in range(array.shape[0]):
            loc = []
            if array [:, l][m] > 0:
                loc.append(m)
                loc.append(l)
                if loc not in contour:
                    contour.append(loc)
                break
        for n in range(array.shape[0]):
            loc = []
            if array [:, l][array.shape[0] - 1 - n] > 0 and (m != (array.shape[0] - 1 - n)):
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
    
    roi2 = roi2[start : (lower+1), end : (right+1)]
    return roi2


def optimal_rotation (roi):
    angle = 0
    degree = []

    while angle < 181:
        degree.append (angle)
        angle += 5
    
    contour_mask = find_contour(roi)
    crop_mask = crop (roi, contour_mask)
    width_min = min (crop_mask.shape[0], crop_mask.shape[1])

    best_angle = 0
    for k in degree:
        ro = rotate(crop_mask, k, resize=True)
        contour_ro = find_contour(ro)
        crop_ro = crop (ro, contour_ro)
        ro_width = min (crop_ro.shape[0], crop_ro.shape[1])
                
        if ro_width < width_min:
            width_min = ro_width
            best_angle = k
            
    if best_angle == 0:
        rotated_roi = crop_mask
    else:
        rotated_roi = rotate(crop_mask, best_angle, resize=True)
        contour_ro = find_contour(rotated_roi)
        crop_ro = crop (rotated_roi, contour_ro)
        
        if crop_ro.shape[0] > crop_ro.shape[1]:
            rotated_roi = rotate(crop_mask, best_angle + 90, resize=True)    
            
            contour_roi = find_contour(rotated_roi)
            rotated_roi = crop (rotated_roi, contour_roi)    
    
    return rotated_roi


def resizing(roi):  # return roi resized to (350, 600)
    roi2 = copy.deepcopy(roi)
    if (roi.shape[0] / roi.shape[1]) < (7/12):
        dims = (150, (roi.shape[0] * 150) // roi.shape[1])
    else:
        dims = ((87 * roi.shape[1]) // roi.shape[0], 87)
    roi2 = cv2.resize(roi2, dsize=dims, interpolation=cv2.INTER_NEAREST)
    
    shape_diff = np.array((87, 150)) - np.array(roi2.shape)
    new_roi = np.lib.pad(roi2, ((0,shape_diff[0]),(0,shape_diff[1])), 
                             'constant', constant_values=(0))
    return new_roi

def exp_value(array_2D):  # return exponential array
    exp = np.multiply(array_2D, array_2D)
    return exp

def border_medulla(roi):# return roi for no border, medulla for medulla area
    contour = find_contour(roi)
    medulla = copy.deepcopy(roi)
    
    upper = contour[0][0]
    lower = upper

    for i in contour:
        if i[0] > lower:
            lower = i[0]
        if i[0] < upper:
            upper = i[0]
            
    width_squared = (lower - upper +1) **2
    
    for con in contour:
        for i in range(roi.shape[0]):
            for j in range(roi.shape[1]):
                temp = (i - con[0])**2 + (j - con[1])**2
                if temp > 0 and (temp <= (width_squared // 400)):  # 10% of width for cutting border
                    roi[i][j] = 0
                if temp > 0 and (temp <= (width_squared * 49 // 400 )):  # 35% of width for medulla
                    medulla[i][j] = 0
    return roi, medulla


def stadardization (roi, med):
    mean_roi = np.mean (roi[roi>0])
    mean = np.mean (med[med>0])
    std = np.std (med[med>0])
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            if roi[i][j] > 230*230:  ### remove markers
                roi[i][j] = mean_roi
    roi2 = (roi - mean)/ std
    return roi2


a, b =[], []
images = os.listdir('D:/images_png_VN/')
for i in images:
    a.append ('D:/images_png_VN/' + i)
bugs = []    

labels = os.listdir('D:/labels_png_VN/')
for j in labels:
    b.append ('D:/labels_png_VN/' + j)

for i in range(len (a)):
    start = timeit.default_timer()
    
    seg = segmentation (a[i], b[i])
    
    ro = optimal_rotation (seg)
    
    resized = resizing (ro)

    ROI, medulla = border_medulla(resized)
    
    if np.mean (medulla) == 0:
        bugs.append (a[i])
        print ('goddamn bug over here')
        
    save ('D:/Pre-processing/VN_png_roi_medulla_wo_exp_&_std/roi%s' %a[i][18:-4] + '_' + '%s.npy' %i, ROI)
    save ('D:/Pre-processing/VN_png_roi_medulla_wo_exp_&_std/medulla%s' %a[i][18:-4] + '_' + '%s.npy' %i, medulla)    
    
    stop = timeit.default_timer()
    
    print ('FINAL STEPPPPP of %s' %i, 'with time:', stop-start)

# VN_com = os.listdir ('D:/ext_VN_complete/')
# images = os.listdir('D:/labels/')

# for i in VN_com:
#     for j in images:
#         if os.path.splitext(i[2:-4])[0] == j[2:-4]:
#             scr = 'D:/labels/' + j
#             dst = 'D:/labels_png_VN/%s' %i[0] + '_' + j[:-4] +'.png'
#             shutil.copyfile (scr, dst)
