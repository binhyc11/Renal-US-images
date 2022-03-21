import os, random, wandb
import numpy as np
# from sklearn.model_selection import train_test_split
from tensorflow import keras
# from matplotlib import pyplot
from keras.models import  Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Input, Concatenate, GlobalAveragePooling2D
# from keras.preprocessing.image import ImageDataGenerator, random_shift, random_rotation
# from keras.applications.vgg19 import VGG19
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint
from skimage.transform import rotate

wandb.init (project = "RenalUS", entity="binhyc11")
model_callback = ModelCheckpoint (filepath='D:/models/branch_size_aug_com.h5', save_weights_only=False,
                                    monitor='val_accuracy', mode='max', save_best_only=True)

batch_size = 128
lr_rate = 1e-6
epoch = 30000

def create_dataset (path):
    X_img_left, X_img_right, X_size_left, X_size_right, y = [],[], [], [], []
    
    if 'train' in path:
        for i in os.listdir (path + '0/'):
            img = np.load(path + '0/' + i)
            
            left = img[:,:,0]
            X_img_left.append (left)
            X_img_left.append (augmentation(left))
            
            right = img[:,:,1]
            X_img_right.append (right)
            X_img_right.append (augmentation(right))
            
            for j in info:
                if j[0][:-2] == i[:len (j[0][:-2])]:
                    if 'L' in j[0]:
                        X_size_left.append ([float (j[1])])
                        X_size_left.append ([float (j[1])])
                    if 'R' in j[0]:
                        X_size_right.append ([float (j[1])])
                        X_size_right.append ([float (j[1])])
            
            y.append ([1., 0.])
            y.append ([1., 0.])
        
    else:
        for i in os.listdir (path + '0/'):
            img = np.load(path + '0/' + i)
            
            left = img[:,:,0]
            X_img_left.append (left)
            
            right = img[:,:,1]
            X_img_right.append (right)
            
            for j in info:
                if j[0][:-2] == i[:len (j[0][:-2])]:
                    if 'L' in j[0]:
                        X_size_left.append ([float (j[1])])
                    if 'R' in j[0]:
                        X_size_right.append ([float (j[1])])
            
            y.append ([1., 0.])
        
    for i in os.listdir (path + '1/'):
        img = np.load(path + '1/' + i)
        
        left = img[:,:,0]
        X_img_left.append (left)
        
        right = img[:,:,1]
        X_img_right.append (right)
        
        for j in info:
            if j[0][:-2] == i[:len (j[0][:-2])]:
                if 'L' in j[0]:
                    X_size_left.append ([float (j[1])])
                if 'R' in j[0]:
                    X_size_right.append ([float (j[1])])
                
        y.append ([0., 1.])
    
    X_img_left, X_img_right      =  np.array(X_img_left),  np.array(X_img_right)
    X_size_left, X_size_right, y =  np.array(X_size_left),  np.array(X_size_right),  np.array(y)
    
    return np.expand_dims(X_img_left, axis=-1) , np.expand_dims(X_size_left, axis=-1) , np.expand_dims(X_img_right, axis=-1) , np.expand_dims(X_size_right, axis=-1), y

def augmentation (image):
    # shifted = random_shift (image, wrg = 0.1, hrg = 0.1, row_axis=0, col_axis=1, channel_axis=2, fill_mode='nearest')
    angle = random.uniform(-10, 10)
    rotated = rotate (image, angle, resize=False)
    return  rotated



def define_branch_model():
    ## left branch
    inputs_img_left = Input(shape = (87,150,1), name='img_left')
    img_left = Conv2D(32, 7, activation='relu',
                      kernel_initializer= 'he_uniform',
                      padding = 'same')(inputs_img_left)
    img_left = MaxPooling2D(3)(img_left)
    
    img_left = Conv2D(64, 5, activation='relu', padding = 'same')(img_left)
    img_left = MaxPooling2D(3)(img_left)
    
    img_left = Conv2D(128, 3, activation='relu', padding = 'same')(img_left)
    img_left = MaxPooling2D(3)(img_left)
    
    img_left = Flatten()(img_left)
    # img_left = Dropout(0.5)(img_left)
    
    img_left = Dense(256, activation= 'relu')(img_left)
    img_left = Dropout(0.5)(img_left)
    
    img_left = Dense(4, activation= 'relu')(img_left)
    

    inputs_size_left = Input(shape = (1,1), name='size_left')
    
    size_left = Flatten()(inputs_size_left)
    
    size_left = Dense(4, activation= 'relu')(size_left)
      
    left = Concatenate()([img_left, size_left])
    
    ## right branch
    inputs_img_right = Input(shape = (87,150,1), name='img_right')
    img_right = Conv2D(32, 7, activation='relu',
                      kernel_initializer= 'he_uniform',
                      padding = 'same')(inputs_img_right)
    img_right = MaxPooling2D(3)(img_right)
    
    img_right = Conv2D(64, 5, activation='relu', padding = 'same')(img_right)
    img_right = MaxPooling2D(3)(img_right)
    
    img_right = Conv2D(128, 3, activation='relu', padding = 'same')(img_right)
    img_right = MaxPooling2D(3)(img_right)
    
    img_right = Flatten()(img_right)
    # img_right = Dropout(0.5)(img_right)
    
    img_right = Dense(256, activation= 'relu')(img_right)
    img_right = Dropout(0.5)(img_right)
    
    img_right = Dense(4, activation= 'relu')(img_right)
    

    inputs_size_right = Input(shape = (1,1), name='size_right')
    
    size_right = Flatten()(inputs_size_right)
    
    size_right = Dense(4, activation= 'relu')(size_right)
      
    right = Concatenate()([img_right, size_right])
    
    
    output = Concatenate()([left, right])
    output = Dense(2, activation='softmax', name='output')(output)
    
    model = Model (inputs = [inputs_img_left, inputs_size_left, inputs_img_right, inputs_size_right], outputs = output)
    opt = keras.optimizers.Adam (learning_rate = lr_rate)
    model.compile(optimizer=opt, loss = 'binary_crossentropy', metrics=['accuracy'])
    
    
    return model
    
model = define_branch_model()
model.summary()


info = np.load ('D:/v4.npy')

X_img_left_train, X_size_left_train, X_img_right_train, X_size_right_train, y_train = create_dataset ('D:/train_co/')
X_img_left_val  , X_size_left_val  , X_img_right_val  , X_size_right_val  , y_val   = create_dataset ('D:/val_co/')
X_img_left_test , X_size_left_test , X_img_right_test , X_size_right_test , y_test  = create_dataset ('D:/test_co/')




history = model.fit(
                    [X_img_left_train, X_size_left_train, X_img_right_train, X_size_right_train],
                    y_train,
                    batch_size = batch_size,
                    epochs = epoch,
                    callbacks = [WandbCallback(),  model_callback],
                    validation_data = ([X_img_left_val  , X_size_left_val, X_img_right_val  , X_size_right_val], y_val),
                    steps_per_epoch = len (y_train)// batch_size,
                    validation_steps = len(y_val) // batch_size,
                    verbose = 1,
                    shuffle=True
                    )
