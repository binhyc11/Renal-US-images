import os, shutil, wandb
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from matplotlib import pyplot
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator, random_shift, random_rotation
from keras.applications.vgg19 import VGG19
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint


wandb.init (project = "RenalUS", entity="binhyc11")
model_callback = ModelCheckpoint (filepath='D:/models/vgg19_aug_com.h5', save_weights_only=False,
                                    monitor='val_loss', mode='min', save_best_only=True)

batch_size = 128
lr_rate = 1e-6
epoch = 10000

def create_dataset (path):
    X, y = [],[]
    if 'train' in path:
        for i in os.listdir (path + '0/'):
            img = np.dstack((np.load(path + '0/' + i), np.zeros((100, 140))))
            X.append (img)
            y.append ([1., 0.])
            X.append (augmentation(img))
            y.append ([1., 0.])
     

        for i in os.listdir (path + '1/'):
            X.append (np.dstack((np.load(path + '1/' + i), np.zeros((100, 140)))))
            y.append ([0., 1.])

    else:
        for i in os.listdir (path + '0/'):
            X.append (np.dstack((np.load(path + '0/' + i), np.zeros((100, 140)))))
            y.append ([1., 0.])

        for i in os.listdir (path + '1/'):
            X.append (np.dstack((np.load(path + '1/' + i), np.zeros((100, 140)))))
            y.append ([0., 1.])

    X =   np.array(X)
    y =   np.array(y)
    
    return X, y

def augmentation (image):
    shifted = random_shift (image, wrg = 0.1, hrg = 0.1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    return  shifted


def define_model():
    model = Sequential()
    model.add (Conv2D(256, (7, 7), activation='relu',
                      kernel_initializer= 'he_uniform',
                      padding = 'same',
                      input_shape = (100, 140, 2)
                      ))
    model.add (MaxPooling2D(2, 2))

    model.add (Conv2D(128, (7, 7), activation='relu'))
    model.add (MaxPooling2D(2, 2))

    model.add (Conv2D(64, (5, 5), activation='relu'))
    model.add (MaxPooling2D(2, 2))

    model.add (Conv2D(128, (3, 3), activation='relu'))
    model.add (MaxPooling2D(2, 2))

    model.add (Flatten())
    model.add (Dense(128, activation= 'relu', kernel_initializer= 'he_uniform'))
    
    model.add(Dropout(0.5))
    
    model.add (Dense(2, activation= 'softmax'))
    opt = keras.optimizers.Adam (learning_rate = lr_rate)
    model.compile(optimizer=opt, loss = 'binary_crossentropy', 
                  metrics=['accuracy'])
    return model

def define_model_vgg19():
    model = VGG19(include_top=False, input_shape=(100, 140, 3), classes=2)
    for layer in model.layers:
        layer.trainable = False
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = 'relu', kernel_initializer='he_uniform')(flat1)
    drop = Dropout(0.3) (class1)
    output = Dense (2, activation = 'softmax')(drop)
    model = Model(inputs = model.inputs, outputs = output)
    opt = keras.optimizers.Adam (learning_rate = lr_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


model = define_model_vgg19()
model.summary()


X_train, y_train = create_dataset ('D:/train/')
X_val, y_val = create_dataset ('D:/val/')
X_test, y_test = create_dataset ('D:/test/')

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


    

