import os, random, wandb
import numpy as np
# from sklearn.model_selection import train_test_split
from tensorflow import keras
# from matplotlib import pyplot
from keras.models import  Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Concatenate
# from keras.preprocessing.image import ImageDataGenerator, random_shift, random_rotation
# from keras.applications.vgg19 import VGG19
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint
from skimage.transform import rotate

wandb.init (project = "RenalUS", entity="binhyc11")
model_callback = ModelCheckpoint (filepath='D:/models/branch_aug_com.h5', save_weights_only=False,
                                    monitor='val_loss', mode='min', save_best_only=True)

batch_size = 128
lr_rate = 1e-6
epoch = 10000

def create_dataset (path):
    X_left, X_right, y = [],[], []
    
    if 'train' in path:
        for i in os.listdir (path + '0/'):
            img = np.load(path + '0/' + i)
            left = img[:, :, 0]
            right = img[:, :, 1]
            
            X_left.append (left)
            X_right.append (right)
            y.append ([1., 0.])
            
            X_left.append (augmentation(left))
            X_right.append (augmentation(right))
            y.append ([1., 0.])
        
    else:
        for i in os.listdir (path + '0/'):
            img = np.load(path + '0/' + i)
            left = img[:, :, 0]
            right = img[:, :, 1]
            
            X_left.append (left)
            X_right.append (right)
            y.append ([1., 0.])
        
    for i in os.listdir (path + '1/'):
        img = np.load(path + '1/' + i)
        left = img[:, :, 0]
        right = img[:, :, 1]
        
        X_left.append (left)
        X_right.append (right)
        y.append ([0., 1.])
        
    X_left =   np.array(X_left)
    X_right =   np.array(X_right)
    y =   np.array(y)
    
    return np.expand_dims(X_left, axis=-1) , np.expand_dims(X_right, axis=-1) , y

def augmentation (image):
    # shifted = random_shift (image, wrg = 0.1, hrg = 0.1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    angle = random.uniform(-10, 10)
    rotated = rotate (image, angle, resize=False)
    return  rotated




# def define_model():
#     model = Sequential()
#     model.add (Conv2D(256, (7, 7), activation='relu',
#                       kernel_initializer= 'he_uniform',
#                       padding = 'same',
#                       input_shape = (100, 140, 2)
#                       ))
#     model.add (MaxPooling2D(2, 2))

#     model.add (Conv2D(128, (7, 7), activation='relu'))
#     model.add (MaxPooling2D(2, 2))

#     model.add (Conv2D(64, (5, 5), activation='relu'))
#     model.add (MaxPooling2D(2, 2))

#     model.add (Conv2D(128, (3, 3), activation='relu'))
#     model.add (MaxPooling2D(2, 2))

#     model.add (Flatten())
#     model.add (Dense(128, activation= 'relu', kernel_initializer= 'he_uniform'))
    
#     model.add(Dropout(0.5))
    
#     model.add (Dense(2, activation= 'softmax'))
#     opt = keras.optimizers.Adam (learning_rate = lr_rate)
#     model.compile(optimizer=opt, loss = 'binary_crossentropy', 
#                   metrics=['accuracy'])
#     return model

def define_branch_model():
    ## left branch
    inputs_left = Input(shape = (100,140,1), name='left_kiney')
    left = Conv2D(256, 7, activation='relu',
                      kernel_initializer= 'he_uniform',
                      padding = 'same')(inputs_left)
    
    left = MaxPooling2D(3)(left)
    
    left = Conv2D(128, 7, activation='relu')(left)
    left = MaxPooling2D(3)(left)
    
    left = Conv2D(64, 5, activation='relu')(left)
    left = MaxPooling2D(2)(left)
    
    left = Flatten()(left)
    
    ## right branch
    inputs_right = Input(shape = (100,140,1), name='right_kiney')
    right = Conv2D(256, 7, activation='relu',
                      kernel_initializer= 'he_uniform',
                      padding = 'same')(inputs_right)
    right = MaxPooling2D(3)(right)
    
    right = Conv2D(128, 7, activation='relu')(right)
    right = MaxPooling2D(3)(right)
    
    right = Conv2D(64, 5, activation='relu')(right)
    right = MaxPooling2D(2)(right)
    
    right = Flatten()(right)
    
    inputs = Concatenate()([left, right])
    
    output = Dense(128, activation= 'relu', kernel_initializer= 'he_uniform')(inputs)
    output = Dropout(0.5)(output)
    output = Dense(2, activation='softmax', name='output')(output)
    
    model = Model (inputs = [inputs_left, inputs_right], outputs = output)
    opt = keras.optimizers.Adam (learning_rate = lr_rate)
    model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics=['accuracy'])
    
    
    return model
    
model = define_branch_model()
model.summary()

X_left_train, X_right_train, y_train = create_dataset ('D:/train/')
X_left_val, X_right_val, y_val = create_dataset ('D:/val/')
X_left_test, X_right_test, y_test = create_dataset ('D:/test/')




history = model.fit([X_left_train, X_right_train],
          y_train,
          batch_size = batch_size,
          epochs = epoch,
           callbacks = [WandbCallback(),  model_callback],
          validation_data = ([X_left_val, X_right_val], y_val),
          steps_per_epoch = len (y_train)// batch_size,
          validation_steps = len(y_val) // batch_size,
          verbose = 1,
          shuffle=True
          )
