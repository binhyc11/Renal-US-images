import cv2, pydicom, nrrd, os, timeit, copy, csv
import numpy as np
from PIL import Image
from numpy import asarray, save
import matplotlib.pyplot as plt
from skimage.transform import rotate

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
                path_nrrd = directory + ID + '/' + file
                name_dcm = f_name[5:] + '.dcm'
                if name_dcm in files:
                    path_dcm = directory + ID + '/' + name_dcm
                    nrrd_list.append(path_nrrd)
                    dcm_list.append(path_dcm)
    return dcm_list, nrrd_list


def segmentation(path_dcm, path_nrrd):  # return ROI, mask_array
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
    mask_array = np.reshape(mask_array, (mask_array.shape[0], mask_array.shape[1]))
    mask_array = np.transpose(mask_array).astype (float)

    # apply nrrd to dcm in array type
    ROI = np.multiply(mask_array, array_2D).astype (float)

    return ROI, mask_array


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


def overview(area):  # return mean_value_pixel
    num_pixel = 0
    total_value_pixel = 0
    for i in range(area.shape[0]):
        for j in range(area.shape[1]):
            if area[i][j] > 0:
                num_pixel += 1
                total_value_pixel += area[i][j]
    return num_pixel, total_value_pixel

root = 'D:/renalUS/data/'
a, b = path(root)

angle = 5
degree = []
angle_width_path =[]

while angle < 181:
    degree.append (angle)
    angle += 5

for i in range (len (a)):
    
    _, mask = segmentation (a[i], b[i])
    
    contour_mask = find_contour(mask)
    crop_mask = crop (mask, contour_mask)
    ori_value,_ = overview(crop_mask)
    ori_mean = ori_value / (crop_mask.shape[0] * crop_mask.shape[1])
    max_mean = ori_mean
    best_angle = []
    
    
    for k in degree:
        ro = rotate(crop_mask, k, resize=True)
        contour_ro = find_contour(ro)
        crop_ro = crop (ro, contour_ro)
        ro_value, _ = overview (crop_ro)
        ro_mean = ori_value/ (crop_ro.shape[0] * crop_ro.shape[1])
        
        if ro_mean > max_mean:
            temp = []
            temp.append (k)
            temp.append(crop_ro)
            best_angle.append (temp)
    
    width = min (best_angle[-1][1].shape[0], best_angle[-1][1].shape[1])
    
    angle_width_path.append (best_angle[-1][0])
    angle_width_path.append (width)
    angle_width_path.append (a[i])
    
    plt.imshow (best_angle[-1][1])
    plt.show()
    
    print ('#############################')
print (len (angle_width_path))

header = ['Angle', 'Width', 'Path']
with open ('C:/Users/binhy/Desktop/angle_width_path.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(angle_width_path)
