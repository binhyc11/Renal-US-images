import os, copy, wandb
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import random_rotation, random_zoom, random_shift
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint


root = 'D:/Pre-processing/data_roi_medulla_wo_exp_&_std/'

wandb.init (project = "RenalUS", entity="binhyc11")
model_callback = ModelCheckpoint (filepath='D:/sof_7x32_5x64_3x128_lr_9.h5', save_weights_only=False,
                                    monitor='val_loss', mode='min', save_best_only=True)

batch_size = 128
lr_rate = 1e-6
epoch = 5000

def augmentation (image):
    rotated = random_rotation (image, 20, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    shifted = random_shift (image, wrg = 0.1, hrg = 0.1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    return rotated, shifted


def exp_value(array_2D):  # return exponential array
    exp = np.multiply(array_2D, array_2D)
    return exp


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

def create_dataset (path):
    path_5, path_6 = [], []
    for file in os.listdir (path):
        if file[:6] == 'roi_5_':
            path_5.append(file)
        if file [:6] == 'roi_6_':
            path_6.append(file)
    
    tr_5, test_5 = train_test_split(path_5, test_size=0.1, random_state = 1)
    tr_6, test_6 = train_test_split(path_6, test_size=0.1, random_state = 1)
    
    train_5, val_5 = train_test_split(tr_5, test_size=0.2, random_state = 1)
    train_6, val_6 = train_test_split(tr_6, test_size=0.2, random_state = 1)
    
    X_train, X_val, X_test, y_train, y_val, y_test = [],[],[],[],[],[]
    
    for i in train_5:
        
        med = np.load(path + 'medulla' + i[3:])
        roi = np.load(path + i)
        
        SD_roi = remove_outliers(roi, med)        
        exp_roi = exp_value (SD_roi)       
        scale_roi = scale (exp_roi)
        
        img = np.expand_dims(scale_roi, axis=-1)
        
        ro, sh = augmentation (img)        
        X_train.extend((img, ro, sh))
        y_train.extend(([1.,0.], [1.,0.], [1.,0.]))
        print ('I am here')
        
    for i in train_6:
        med = np.load(path + 'medulla' + i[3:])
        roi = np.load(path + i)
        
        SD_roi = remove_outliers(roi, med)
        exp_roi = exp_value (SD_roi)
        scale_roi = scale (exp_roi)
        
        img = np.expand_dims(scale_roi, axis=-1)
       
        ro, sh = augmentation (img)        
        X_train.extend ((img, ro, sh))
        y_train.extend (([0., 1.],[0., 1.],[0., 1.]))
        
    
    for i in val_5:
        med = np.load(path + 'medulla' + i[3:])
        roi = np.load(path + i)
        
        SD_roi = remove_outliers(roi, med)
        exp_roi = exp_value (SD_roi)
        scale_roi = scale (exp_roi)
        
        img = np.expand_dims(scale_roi, axis=-1)
        X_val.append(img)
        y_val.append ([1., 0.])
        
    for i in val_6:
        med = np.load(path + 'medulla' + i[3:])
        roi = np.load(path + i)
        
        SD_roi = remove_outliers(roi, med)
        exp_roi = exp_value (SD_roi)
        scale_roi = scale (exp_roi)
        
        img = np.expand_dims(scale_roi, axis=-1)
        X_val.append(img)
        y_val.append ([0., 1.])
        
        
    for i in test_5:
        med = np.load(path + 'medulla' + i[3:])
        roi = np.load(path + i)
        
        SD_roi = remove_outliers(roi, med)
        exp_roi = exp_value (SD_roi)
        scale_roi = scale (exp_roi)
        
        img = np.expand_dims(scale_roi, axis=-1)
        X_test.append(img)
        y_test.append ([1., 0.])
        
    for i in test_6:
        med = np.load(path + 'medulla' + i[3:])
        roi = np.load(path + i)
        
        SD_roi = remove_outliers(roi, med)
        exp_roi = exp_value (SD_roi)
        scale_roi = scale (exp_roi)
        
        img = np.expand_dims(scale_roi, axis=-1)
        X_test.append(img)
        y_test.append ([0., 1.])
    
    X_train =   np.array(X_train)
    X_val =     np.array(X_val)
    X_test =    np.array(X_test)
    y_train =   np.array(y_train)
    y_val =     np.array(y_val)
    y_test =    np.array(y_test)
    
    # np.save ('D:/X_train.npy', X_train)
    # np.save ('D:/X_val.npy', X_val)
    # np.save ('D:/X_test.npy', X_test)
    # np.save ('D:/y_train.npy', y_train)
    # np.save ('D:/y_val.npy', y_val)
    # np.save ('D:/y_test.npy', y_test)
    
    
    
    return X_train, X_val, X_test, y_train, y_val, y_test



def define_model():
    model = Sequential()
    model.add (Conv2D(32, (7, 7), activation='relu',
                      kernel_initializer= 'he_uniform',
                      padding = 'same',
                      input_shape = (87, 150, 1)
                      ))
    model.add (MaxPooling2D(2, 2))

    model.add (Conv2D(64, (5,5), activation='relu'))
    model.add (MaxPooling2D(2, 2))
    
    model.add (Conv2D(128, (3,3), activation='relu'))
    model.add (MaxPooling2D(2, 2))
    
    model.add (Conv2D(128, (3,3), activation='relu'))
    model.add (MaxPooling2D(2, 2))
    
    model.add (Flatten())
    model.add (Dense(128, activation= 'relu',
                      kernel_initializer= 'he_uniform'))
    
    model.add(Dropout(0.5))
    
    model.add (Dense(2, activation= 'softmax'))
    opt = keras.optimizers.Adam (learning_rate = lr_rate)
    model.compile(optimizer=opt, loss = 'binary_crossentropy', 
                  metrics=['accuracy'])
    return model



def run_test(root):
    model = define_model()
    model.summary()
    X_train, X_val, X_test, y_train, y_val, y_test =  np.load ('D:/X_train.npy'), np.load ('D:/X_val.npy'), np.load ('D:/X_test.npy'), np.load ('D:/y_train.npy'), np.load ('D:/y_val.npy'), np.load ('D:/y_test.npy')
    history = model.fit(X_train,
                        y_train,
                        batch_size = batch_size,
                        epochs = epoch,
                        callbacks = [WandbCallback(),  model_callback],
                        validation_data = (X_val, y_val),
                        steps_per_epoch = len (X_train)// batch_size,
                        validation_steps = len(X_val) // batch_size,
                        verbose = 1,
                        shuffle=True
                        )
    
    return history
run_test(root)
