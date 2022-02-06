from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

def prediction (path):    
    model = load_model("D:/2_layers_best.h5")
    model.compile (loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    img = image.load_img(path, target_size=(195, 334))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict(images)
    return (classes[0][0])
    

list_ext = os.listdir('D:/ext_heather/')

result = []

for i in list_ext:
    path = 'D:/ext_heather/' + str(i)
    pred = prediction(path)
    print (pred)
    temp = []
    temp.append (i)
    temp.append (pred)
    result.append(temp)
