from tensorflow import keras
from matplotlib import pyplot
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
import wandb
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint

wandb.init (project = 'RenalUS', entity = 'binhyc11')
model_callback = ModelCheckpoint (filepath='D:/models/VGG19_2.h5', save_weights_only=False,
                                  monitor='val_loss', mode='min', save_best_only=True)

batch_size = 128
lr_rate = 1e-6
epoch = 5000

    
def define_model_vgg19():
    model = VGG19(include_top=False, input_shape=(194, 334, 3), classes=2)
    for layer in model.layers:
        layer.trainable = False
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation = 'relu', kernel_initializer='he_uniform')(flat1)
    drop = Dropout(0.3) (class1)
    output = Dense (1, activation = 'sigmoid')(drop)
    model = Model(inputs = model.inputs, outputs = output)
    opt = keras.optimizers.Adam (learning_rate = lr_rate)
    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model



def summarize_diagnostics (history):
    pyplot.subplot(211)
    pyplot.title('Loss')
    pyplot.plot(history.history['loss'], color = 'blue', label = 'train')
    pyplot.plot(history.history['val_loss'], color = 'red', label = 'test')
    pyplot.subplot(212)
    pyplot.title('Accuracy')
    pyplot.plot(history.history['accuracy'], color = 'blue', label = 'train')
    pyplot.plot(history.history['val_accuracy'], color = 'red', label = 'test')
    pyplot.show()

def run_test_vgg19():
    model = define_model_vgg19()
    test_datagen = ImageDataGenerator()
    # test_datagen.mean = [123.68, 116.779, 103.939]
    train_datagen = ImageDataGenerator(
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       rotation_range=10,
                                       fill_mode='constant', cval=0.0,
                                       # zoom_range=0.2
                                       )
    # train_datagen.mean = [123.68, 116.779, 103.939]
    train_it = train_datagen.flow_from_directory('D:/train/',
                                           class_mode='binary',
                                           batch_size=batch_size,
                                           target_size=(194,334)
                                           )
    test_it = test_datagen.flow_from_directory('D:/val/',
                                           class_mode='binary',
                                           batch_size=batch_size,
                                           target_size=(194,334)
                                           ) 
    history = model.fit(train_it, steps_per_epoch=(len(train_it)),
                                  validation_data=test_it,
                                  validation_steps = len(test_it),
                                  epochs = epoch,
                                  callbacks=[WandbCallback(),  model_callback],
                                  )
    _, acc = model.evaluate(test_it, steps=len(test_it), verbose =0)
    print ('%.3f' % (acc * 100.0))
    # summarize_diagnostics(history)

run_test_vgg19()
