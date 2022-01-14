import cv2, pydicom, nrrd, os, itertools
import numpy as np
from PIL import Image
from numpy import asarray
from matplotlib import pyplot as plt

root = 'D:/renalUS/data/'

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
    
    # show the segmentation
    segmented = ROI/ ROI.max() * 255.0
    plt.imshow(segmented, interpolation = 'none')
    plt.show()
    return segmented

col =[]
for i in range (800):
    col.append('c' + str(i+1))
    

def plot_pixel_value (row):
    sum = np.sum(row, dtype = np.float32)
    print (sum)
    if sum != 0:
        plt.bar (col, row, color ='maroon')
        plt.ylim((0, 255))
        plt.show()
        
# Total = []
# for i in range (600):
#     sum = np.sum(seg[i], dtype = np.float32)
    
        
        
        
        