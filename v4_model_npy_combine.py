import os, shutil, wandb
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from matplotlib import pyplot
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint


wandb.init (project = "RenalUS", entity="binhyc11")
model_callback = ModelCheckpoint (filepath='D:/models/7x32_5x64_lr5_Dropout0.5_com.h5', save_weights_only=False,
                                  monitor='val_loss', mode='min', save_best_only=True)

batch_size = 256
lr_rate = 1e-5
epoch = 10000

def create_dataset (path):
    X, y = [],[]
    
    for i in os.listdir (path + '0/'):
        X.append (np.load(path + '0/' + i))
        y.append ([1., 0.])
    
    for i in os.listdir (path + '1/'):
        X.append (np.load(path + '1/' + i))
        y.append ([0., 1.])
        
    X =   np.array(X)
    y =   np.array(y)
    
    return X, y

def define_model():
    model = Sequential()
    model.add (Conv2D(32, (7, 7), activation='relu',
                      kernel_initializer= 'he_uniform',
                      padding = 'same',
                      input_shape = (100, 140, 2)
                      ))
    model.add (MaxPooling2D(2, 2))

    model.add (Conv2D(64, (5, 5), activation='relu'))
    model.add (MaxPooling2D(2, 2))

    model.add (Flatten())
    model.add (Dense(128, activation= 'relu', kernel_initializer= 'he_uniform'))
    
    model.add(Dropout(0.5))
    
    model.add (Dense(2, activation= 'softmax'))
    opt = keras.optimizers.Adam (learning_rate = lr_rate)
    model.compile(optimizer=opt, loss = 'binary_crossentropy', 
                  metrics=['accuracy'])
    return model



model = define_model()
model.summary()
keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)

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
# model.summary()

    

