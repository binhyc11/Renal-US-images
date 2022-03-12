import os, random, wandb
import numpy as np
# from sklearn.model_selection import train_test_split
from tensorflow import keras
# from matplotlib import pyplot
from keras.models import  Model, Sequential
from keras.layers import Conv2D, MaxPooling2D,SeparableConv2D, Dense, Flatten, Dropout, Input, Concatenate, BatchNormalization, Activation
# from keras.preprocessing.image import ImageDataGenerator, random_shift, random_rotation
# from keras.applications.vgg19 import VGG19
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint
from skimage.transform import rotate
from keras import layers

wandb.init (project = "RenalUS", entity="binhyc11")
model_callback = ModelCheckpoint (filepath='D:/models/Xceoption_branch_aug_com.h5', save_weights_only=False,
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

def branch(inputs):

    # Entry block
    
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    for size in [128, 256, 728]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual
    
    # Exit block 
    residual = layers.Conv2D(1024, 1, strides=2, padding="same")(
            previous_block_activation
        )
    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(728, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Activation("relu")(x)
    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)
    
    x = layers.add([x, residual])  # Add back residual

        
    x = layers.SeparableConv2D(1536, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.SeparableConv2D(2048, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    
    x = layers.GlobalAveragePooling2D()(x)
    
    x = layers.Flatten() (x)
       
    return x

def branch_Xception():
    inputs_left = Input(shape = (100,140,1), name='left_kiney')
    left = branch(inputs_left)
    
    inputs_right = Input(shape = (100,140,1), name='right_kiney')
    right = branch(inputs_right)
    
    inputs = Concatenate()([left, right])
    
    output = Dense(256, kernel_regularizer='l2', activation= 'relu', kernel_initializer= 'he_uniform')(inputs)
    output = Dropout(0.5)(output)
    output = Dense(2, kernel_regularizer='l2', activation='softmax', name='output')(output)
    
    model = Model (inputs = [inputs_left, inputs_right], outputs = output)
    opt = keras.optimizers.Adam (learning_rate = lr_rate)
    model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics=['accuracy'])
    
    return model
    
model = branch_Xception()
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
