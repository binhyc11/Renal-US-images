import numpy as np
from PIL import Image
from numpy import asarray, save
import matplotlib.pyplot as plt

def path(directory):  # return dcm_list, nrrd_list
	@@ -24,7 +25,8 @@ def path(directory):  # return dcm_list, nrrd_list
                    dcm_list.append(path_dcm)
    return dcm_list, nrrd_list

def segmentation(path_dcm, path_nrrd):  # return ROI, mask_array
    '''
    Input: path of .dcm file and according path of .nrrd file
    Return: ROI: segmentation of the kidney with border
	@@ -49,22 +51,9 @@ def segmentation(path_dcm, path_nrrd):  # return ROI, mask_array
    # apply nrrd to dcm in array type
    ROI = np.multiply(mask_array, array_2D).astype (float)

    return ROI, mask_array

def delete_markers(ROI):  # return roi without markers
    '''
    Input: ROI: the segmentaion of the kidney with border and markers
    Return: ROI: the segmentation with markers having mean pixel values 
    Note: require overview function
    '''
    num, total = overview(ROI)
    mean = total/num
    for i in range(ROI.shape[0]):
        for j in range(ROI.shape[1]):
            if ROI[i][j] >= 230:
                ROI[i][j] = mean
    return ROI

def find_contour(array):  # return contour of an array
    '''
    Input: a array of segmentation with border
	@@ -106,37 +95,6 @@ def find_contour(array):  # return contour of an array
                break           
    return contour

def border_medulla(contour, roi):# return roi for no border, medulla for medulla area
    '''
    Input:  roi: a (600, 800) array contains segmentation of the kidney
            contour: of roi
    Return: roi: roi from Input with border cut
            medulla: a (600, 800) array contains medulla
    '''
    upper = contour[0][0]
    lower = upper
    left = contour[0][1]
    right = left
    for i in range(len(contour)):
        if contour[i][1] > right:
            right = contour[i][1]
        if contour[i][1] < left:
            left = contour[i][1]
        if contour[i][0] > lower:
            lower = contour[i][0]
        if contour[i][0] < upper:
            upper = contour[i][0]

    medulla = copy.deepcopy(roi)
    for con in contour:
        for i in range(upper, lower + 1):
            for j in range(left, right +1):
                temp = (i - con[0])**2 + (j - con[1])**2
                if temp > 0 and (temp <= 900):  # for cutting border
                    roi[i][j] = 0
                if temp > 0 and (temp <= 10000):  # for medulla
                    medulla[i][j] = 0
    return roi, medulla

def crop(roi, contour):  # return rectangle contains only roi area
    upper = contour[0][0]
	@@ -153,85 +111,122 @@ def crop(roi, contour):  # return rectangle contains only roi area
            lower = contour[i][0]
        if contour[i][0] < upper:
            upper = contour[i][0]
    roi2 = roi2[(upper-1):(lower+1), (left-1) : (right+1)]
    return roi2

def resizing(roi):  # return roi resized to (350, 600)
    roi2 = copy.deepcopy(roi)    
    if (roi.shape[0] / roi.shape[1]) < (7/12):
        dims = (600, (roi.shape[0] * 600) // roi.shape[1])
    else:
        dims = ((350 * roi.shape[1]) // roi.shape[0], 350)
    roi2 = cv2.resize(roi2, dsize=dims, interpolation=cv2.INTER_NEAREST)
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

def exp_value(array_2D):  # return exponential array
    exp = np.multiply(array_2D, array_2D)
    return exp

root = 'D:/external/'
a, b = path(root)
small = []
new_shape = (350, 600)

for i in range (len(a)): 
        start = timeit.default_timer()

        roi, mask = segmentation(a[i], b[i])
        print ('step %s.1' %i)

        roi_no_marker = delete_markers(roi)
        print ('step %s.2' %i)

        contour_roi_no_marker = find_contour(roi_no_marker)
        print ('step %s.3' %i)

        roi_crop = crop(roi_no_marker, contour_roi_no_marker)
        print ('step %s.4' %i)

        roi_resize = resizing(roi_crop)
        print ('step %s.5' %i)

        contour_roi_resize = find_contour(roi_resize)
        print ('step %s.6' %i)

        roi_no_border, medulla = border_medulla(contour_roi_resize, roi_resize)
        print ('step %s.7' %i)
        plt.imshow(roi_no_border)
        plt.show()
        plt.imshow(medulla)
        plt.show()

        num_pixel, total_pixel_value = overview (medulla)
        if num_pixel == 0:
            small.append(i)
        else:
            roi_nor = roi_resize * 255.0 * total_pixel_value/ num_pixel # normalized roi
            print ('step %s.9' %i)

            roi_exp = np.multiply(roi_nor, roi_nor)
            print ('step %s.10' %i)

            shape_diff = np.array(new_shape) - np.array(roi_exp.shape)
            new_roi_exp = np.lib.pad(roi_exp, ((0,shape_diff[0]),(0,shape_diff[1])), 
                              'constant', constant_values=(0))

            plt.imshow(new_roi_exp, cmap="gray")
            plt.axis('off')
            plt.savefig('D:/ext2/%s' %a[i][12:20] + '_%s.png' %i, bbox_inches='tight',pad_inches = 0)


            stop = timeit.default_timer()
            print ('FINAL STEPPPPP of %s' %i, 'with time:', stop-start)
print (small)
