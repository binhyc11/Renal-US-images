import cv2
import pydicom
import numpy as np
from PIL import Image
from numpy import asarray
import nrrd
from matplotlib import pyplot as plt

# convert dcm --> array --> RGB_jpg --> gray_jpg --> array
dcm_to_array = pydicom.dcmread('D:/l-0.dcm')
array = dcm_to_array.pixel_array.astype(float)
array = (np.maximum(array, 0)/array.max()) * 255.0
array = np.uint8(array)
jpg = Image.fromarray(array)
jpg.save('D:/l-0.jpg')
jpg = cv2.imread ('D:/l-0.jpg')
gray_jpg = cv2.cvtColor(jpg, cv2.COLOR_BGR2GRAY)
array_2D = asarray (gray_jpg)

# nrrd --> array  --> transposing
mask_array, header = nrrd.read("D:/mask-l-0.nrrd")
mask_array = np.reshape (mask_array, (800, 600))
mask_array = np.transpose(mask_array)

# apply nrrd to dcm in array type
ROI = np.multiply(mask_array, array_2D)

# show the segmentation
segmentation = ROI/ ROI.max() * 255.0
plt.imshow(segmentation, interpolation = 'none')
plt.show()