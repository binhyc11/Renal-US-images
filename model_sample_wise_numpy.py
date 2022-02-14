import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import random_rotation, random_zoom, random_shift
import wandb
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint


root = 'D:/Binh/Pre-processing/processed_data_sample_wise/'

wandb.init (project = "RenalUS", entity="wanglee")
model_callback = ModelCheckpoint (filepath='D:/Binh/sof_7x32_5x64_lr_5_Dropout_0.5.h5', save_weights_only=False,
                                    monitor='val_loss', mode='min', save_best_only=True)

batch_size = 64
lr_rate = 1e-8
epoch = 2000

def augmentation (image):
    rotated = random_rotation (image, 20, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    shifted = random_shift (image, wrg = 0.1, hrg = 0.1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    return rotated, shifted


def create_dataset (path):
    path_5, path_6 = [], []
    for file in os.listdir (path):
        if file[:2] == '5_':
            path_5.append(file)
        if file [:2] == '6_':
            path_6.append(file)
    
    tr_5, test_5 = train_test_split(path_5, test_size=0.1, random_state = 1)
    tr_6, test_6 = train_test_split(path_6, test_size=0.1, random_state = 1)
    
    train_5, val_5 = train_test_split(tr_5, test_size=0.2, random_state = 1)
    train_6, val_6 = train_test_split(tr_6, test_size=0.2, random_state = 1)
    
    X_train, X_val, X_test, y_train, y_val, y_test = [],[],[],[],[],[]
    
    for i in train_5:
        img = np.expand_dims(np.load(path + i), axis=-1)
        ro, sh = augmentation (img)        
        X_train.extend((img, ro, sh))
        y_train.extend(([1.,0.], [1.,0.], [1.,0.]))
        
    for i in train_6:
        img = np.expand_dims(np.load(path + i), axis=-1)
        ro, sh = augmentation (img)        
        X_train.extend ((img, ro, sh))
        y_train.extend (([0., 1.],[0., 1.],[0., 1.]))
        
    
    for i in val_5:
        X_val.append(np.expand_dims(img, axis=-1))
        y_val.append ([1., 0.])
        
    for i in val_6:
        X_val.append(np.expand_dims(img, axis=-1))
        y_val.append ([0., 1.])
        
        
    for i in test_5:
        X_test.append(np.expand_dims(img, axis=-1))
        y_test.append ([1., 0.])
        
    for i in test_6:
        X_test.append(np.expand_dims(img, axis=-1))
        y_test.append ([0., 1.])
    
    X_train = np.array(X_train)
    X_val = np.array(X_val)
    X_test =    np.array(X_test)
    y_train = np.array(y_train)
    y_val =     np.array(y_val)
    y_test =   np.array(y_test)
    
    return X_train, X_val, X_test, y_train, y_val, y_test



def define_model():
    model = Sequential()
    model.add (Conv2D(32, (5, 5), activation='relu',
                      kernel_initializer= 'he_uniform',
                      padding = 'same',
                      input_shape = (87, 150, 1)
                      ))
    model.add (MaxPooling2D(2, 2))

    model.add (Conv2D(64, (3,3), activation='relu'))
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
    X_train, X_val, X_test, y_train, y_val, y_test = create_dataset (root)
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
    # model.summary()
    return history
run_test(root)
