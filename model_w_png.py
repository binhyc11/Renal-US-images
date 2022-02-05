from tensorflow import keras
from matplotlib import pyplot
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
import wandb
from wandb.keras import WandbCallback


wandb.init (project = "RenalUS", entity="binhyc11")
model_callback = ModelCheckpoint (filepath='D:/model/2_layers.h5', save_weights_only=False,
                                  monitor='val_loss', mode='min', save_best_only=True)

def define_model():
    model = Sequential()
    model.add (Conv2D(32, (7, 7), activation='relu',
                      kernel_initializer= 'he_uniform',
                      padding = 'same',
                      input_shape = (195, 334, 3)
                      ))
    model.add (MaxPooling2D(2, 2))
    model.add (Flatten())
    model.add (Dense(128, activation= 'relu',
                      kernel_initializer= 'he_uniform'))
    model.add (Dense(1, activation= 'sigmoid'))
    opt = keras.optimizers.Adam (learning_rate = 1e-6)
    model.compile(optimizer=opt, loss = 'binary_crossentropy', 
                  metrics=['accuracy'])
    return model
    
def define_model_vgg19():
    model = VGG19(include_top=False, input_shape=(224, 224, 3))
    for layer in model.layers:
        layer.trainable = False
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = 'relu', kernel_initializer='he_uniform')(flat1)
    output = Dense (1, activation = 'sigmoid')(class1)
    model = Model(inputs = model.inputs, outputs = output)
    opt = keras.optimizers.Adam (learning_rate = 1e-6)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def run_test():
    model = define_model()
    train_datagen = ImageDataGenerator(rescale=1.0/255.0,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       rotation_range=10,
                                       fill_mode='constant', cval=0.0,
                                       zoom_range=0.2
                                       )
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    train_it = train_datagen.flow_from_directory('D:/train/',
                                            class_mode= 'binary',
                                            batch_size = 128,
                                            target_size = (195, 334)
                                            )
    test_it = test_datagen.flow_from_directory('D:/test/',
                                          class_mode= 'binary',
                                          batch_size= 128,
                                          target_size = (195, 334)
                                          )
    history = model.fit(train_it, steps_per_epoch=(len(train_it)),
                                  validation_data=test_it,
                                  validation_steps = len(test_it),
                                  epochs = 500,
                                  callbacks=[WandbCallback(),  model_callback],
                                  )
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose =0)
    print ('%.3f' % (acc * 100.0))
    model.save ('D:/2_layers.h5')


def run_test_vgg19():
    model = define_model_vgg19()
    test_datagen = ImageDataGenerator(featurewise_center=True)
    # test_datagen.mean = [123.68, 116.779, 103.939]
    train_datagen = ImageDataGenerator(featurewise_center=True,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       rotation_range=10,
                                       fill_mode='constant', cval=0.0,
                                       zoom_range=0.2
                                       )
    # train_datagen.mean = [123.68, 116.779, 103.939]
    train_it = train_datagen.flow_from_directory('D:/train/',
                                           class_mode='binary',
                                           batch_size=256,
                                           target_size=(224,224)
                                           )
    test_it = test_datagen.flow_from_directory('D:/test/',
                                           class_mode='binary',
                                           batch_size=256,
                                           target_size=(224,224)
                                           ) 
    history = model.fit(train_it, steps_per_epoch=(len(train_it)),
                                  validation_data=test_it,
                                  validation_steps = len(test_it),
                                  epochs = 500,
                                  callbacks=[WandbCallback(), model_callback]
                                  )
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose =0)
    print ('%.3f' % (acc * 100.0))
    # summarize_diagnostics(history)

run_test()
