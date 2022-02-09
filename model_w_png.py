from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os, csv

def prediction (model, path):    
    
    
    img = image.load_img(path, target_size=(195, 334))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict(images)
    return (classes[0][0])
    

list_ext = os.listdir('D:/data/ext_VN_complete/')

result = []

model = load_model("D:/2_layers_original_best.h5")
model.compile (loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

for i in list_ext:
    path = 'D:/data/ext_VN_complete/' + str(i)
    pred = prediction(model, path)
    print (pred)
    temp = []
    temp.append (i)
    temp.append (pred)
    result.append(temp)
    
with open ('C:/Users/binhy/Desktop/result_VN_complete.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(result)
