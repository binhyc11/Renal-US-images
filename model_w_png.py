import os, shutil, wandb
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
model_callback = ModelCheckpoint (filepath='D:/9x32_7x64_5x128_3x128.h5', save_weights_only=False,
                                  monitor='val_loss', mode='min', save_best_only=True)

batch_size = 128
lr_rate = 1e-6
epoch = 5000

def define_model():
    model = Sequential()
    model.add (Conv2D(32, (9, 9), activation='relu',
                      kernel_initializer= 'he_uniform',
                      padding = 'same',
                      input_shape = (194, 334, 3)
                      ))
    model.add (MaxPooling2D(2, 2))

    model.add (Conv2D(64, (7, 7), activation='relu'))
    model.add (MaxPooling2D(2, 2))
    
    model.add (Conv2D(128, (5, 5), activation='relu'))
    model.add (MaxPooling2D(2, 2))
    
    model.add (Conv2D(128, (3, 3), activation='relu'))
    model.add (MaxPooling2D(2, 2))

    model.add (Flatten())
    model.add (Dense(128, activation= 'relu',
                      kernel_initializer= 'he_uniform'))
    model.add (Dense(2, activation= 'softmax'))
    opt = keras.optimizers.Adam (learning_rate = lr_rate)
    model.compile(optimizer=opt, loss = 'binary_crossentropy', 
                  metrics=['accuracy'])
    return model


def run_test():
    model = define_model()
    model.summary()
    train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                        height_shift_range=0.1,
                                        rotation_range=10,
                                        fill_mode='constant', cval=0.0,
                                        zoom_range=0.2
                                        )
    test_datagen = ImageDataGenerator()
    train_it = train_datagen.flow_from_directory('D:/train/',
                                            batch_size = batch_size,
                                            target_size = (194, 334)
                                            )
    test_it = test_datagen.flow_from_directory('D:/val/',
                                          batch_size= batch_size,
                                          target_size = (194, 334)
                                          )
    history = model.fit(train_it, steps_per_epoch=(len(train_it)),
                                  validation_data=test_it,
                                  validation_steps = len(test_it),
                                  epochs = epoch,
                                  callbacks=[WandbCallback(),  model_callback],
                                  )
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose =0)
    print ('%.3f' % (acc * 100.0))


run_test()



def split ():
    
    root = 'D:/Pre-processing/data_scale/'
    
    path_0, path_1 = [], []
    
    
    for i in os.listdir (root):
        if i[0] == '5':
            path_0.append (root + i)
        else:
            path_1.append (root + i)  
    
    
    train_0, test_5 = train_test_split(path_0, test_size=0.1, random_state = 0)
    train_1, test_6 = train_test_split(path_1, test_size=0.1, random_state = 0)
    
    train_5, val_5 = train_test_split(train_0, test_size=0.2, random_state = 0)
    train_6, val_6 = train_test_split(train_1, test_size=0.2, random_state = 0)
    
    for i in train_5:
        name = os.path.basename (i)
        shutil.copyfile(i, 'D:/train/5/' + name)
    
    for i in test_5:
        name = os.path.basename (i)
        shutil.copyfile(i, 'D:/test/' + name)
    
    for i in train_6:
        name = os.path.basename (i)
        shutil.copyfile(i, 'D:/train/6/' + name)
    
    for i in test_6:
        name = os.path.basename (i)
        shutil.copyfile(i, 'D:/test/' + name)
        
    for i in val_6:
        name = os.path.basename (i)
        shutil.copyfile(i, 'D:/val/6/' + name)
    
    for i in val_5:
        name = os.path.basename (i)
        shutil.copyfile(i, 'D:/val/5/' + name)
