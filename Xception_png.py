import tensorflow as tf
from tensorflow import keras
from keras import layers
import os,wandb
import matplotlib.pyplot as plt
from wandb.keras import WandbCallback
from keras.callbacks import ModelCheckpoint


wandb.init (project = "RenalUS", entity="binhyc11")
model_callback = ModelCheckpoint (filepath='D:/models/Xception_png_9.h5', save_weights_only=False,
                                  monitor='val_loss', mode='min', save_best_only=True)

batch_size = 64
image_size = (194, 334)
lr = 1e-5
epochs = 3000

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "D:/train",
    image_size=image_size,
    batch_size=batch_size,
    label_mode = 'categorical'
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "D:/val",
    image_size=image_size,
    batch_size=batch_size,
    label_mode = 'categorical'
)

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1)
        # layers.RandomWidth(factor = 0.1, interpolation="bilinear"),
        # layers.RandomHeight(factor = 0.1, interpolation="bilinear"),
        # layers.RandomCrop(height = 100, width = 200)
    ]
)

def make_model(input_shape, num_classes):
    inputs = keras.Input(shape=input_shape)
    # Image augmentation block
    x = data_augmentation(inputs)

    # Entry block
    x = layers.Rescaling(1.0 / 255)(x)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
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

    
    # i = 0   
    # # Middle block
    # while i < 8:
    # x = layers.Activation("relu")(x)
    # x = layers.SeparableConv2D(728, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Activation("relu")(x)
    # x = layers.SeparableConv2D(728, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)

    # x = layers.Activation("relu")(x)
    # x = layers.SeparableConv2D(728, 3, padding="same")(x)
    # x = layers.BatchNormalization()(x)
    
    # # Project residual
    # residual = layers.Conv2D(728, 1, strides=1, padding="same")(
    #     previous_block_activation
    # )
    # x = layers.add([x, residual])  # Add back residual
    # previous_block_activation = x  # Set aside next residual
    
    #     i+= 1
    
    
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
    x = layers.Dense(1024, activation= 'relu')(x)
    
    if num_classes == 2:
        activation = "softmax"
        units = 2

    
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs)



model = make_model(input_shape=image_size + (3,) , num_classes=2)
# keras.utils.plot_model(model, show_shapes=True)
print (model.summary())

# callbacks = [
#     keras.callbacks.ModelCheckpoint("D:/Xception_png_2.h5"),
# ]
model.compile(
    optimizer=keras.optimizers.Adam(lr),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.fit(
    train_ds, epochs=epochs, validation_data=val_ds,
    callbacks=[WandbCallback(),  model_callback]
)


# img = keras.preprocessing.image.load_img(
#     "PetImages/Cat/6779.jpg", target_size=image_size
# )
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create batch axis

# predictions = model.predict(img_array)
# score = predictions[0]
# print(
#     "This image is %.2f percent cat and %.2f percent dog."
#     % (100 * (1 - score), 100 * score)
# )
