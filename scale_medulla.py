import os, copy
import numpy as np
import matplotlib.pyplot as plt


root = 'D:/Pre-processing/data_roi_medulla_wo_exp_&_std/'


m = 'D:/Pre-processing/data_roi_medulla_wo_exp_&_std/medulla_01479487_5.npy'
r = 'D:/Pre-processing/data_roi_medulla_wo_exp_&_std/roi_01479487_5.npy'

summary = []

def exp_value(array_2D):  # return exponential array
    exp = np.multiply(array_2D, array_2D)
    return exp


def stadardization(roi, med):
    exp_roi = exp_value(roi)
    exp_med = exp_value(med)
    
    mean = np.mean(exp_med[exp_med > exp_med.min()])
    std = np.std(exp_med[exp_med > exp_med.min()])
                
    std_exp_roi = (exp_roi - mean)/ std

    return std_exp_roi, exp_roi

def remove_outliers (roi, med):
    
    mean = np.mean(med[med > roi.min()])
    std = np.std(med[med > roi.min()])
 
    for i in range(roi.shape[0]):
        for j in range(roi.shape[1]):
            if roi[i][j] > (mean + 2*std):  # remove markers
                roi[i][j] = (mean + 2*std)
    return roi
            
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
    

for i in os.listdir(root):
    if i[0] == 'm' and i[8] in ['5', '6']:
        med = np.load(root + i)
        roi = np.load(root + 'roi' + i[7:])
        
        SD_roi = remove_outliers(roi, med)
        
        # exp_med = exp_value(med)
        exp_roi = exp_value (SD_roi)
        
        scale_roi = scale (exp_roi)
        
        # std_exp_roi, exp_roi = stadardization(roi, med)
        
        # ### remove outliers of std_exp_roi by scaling to maxinum of mean + 2SD
        # SD_roi = remove_outliers (std_exp_roi)
        
        # ### scale to (0, 255)
        # scale_roi = scale (SD_roi)
        

        
        plt.subplot(231)
        plt.hist(roi[roi > 0], bins = 'auto')   #### histogram of roi before exp
        plt.ylim([0,750])
        plt.hist(med[med > 0], bins = 'auto', color = 'orange')   #### histogram of med before exp
        plt.ylim([0,750])
        
        plt.subplot(232)
        plt.imshow(med, cmap = 'gray')          ####  med before exp
        plt.axis('off')
        
        plt.subplot(233)
        plt.imshow(roi, cmap = 'gray')         ####  roi before exp
        plt.axis('off')
        plt.title('ori')
        
         
        plt.subplot(234)
        plt.hist(exp_roi[exp_roi > 0], bins = 'auto')   #### histogram of roi after exp
        plt.yticks(color='w')
        plt.ylim([0,750])
        
        plt.subplot(235)
        plt.hist(scale_roi[scale_roi > scale_roi.min()], bins = 'auto')  #### histogram of roi after scale
        plt.yticks(color='w')
        plt.ylim([0,750])
        plt.title('%s' %i[8:-4], loc = 'center'  )
        
        plt.subplot(236)
        plt.imshow(scale_roi, cmap = 'gray')   #### roi after scale
        plt.axis('off')
        plt.title('scale')
        
        
        plt.savefig('D:/graph_scale_medulla/%s' %i[8:-4] + '.png' , bbox_inches='tight')
        plt.close()
        print ('I am running at step %s' %i)
