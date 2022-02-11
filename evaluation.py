from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os, csv

def prediction (path, model):    
    img = image.load_img(path, target_size=(195, 334))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    
    images = np.vstack([x])
    classes = model.predict(images)
    return (classes[0])

    
model = load_model("D:/models/2_layers_original_best.h5")
model.compile (loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

list_ext = os.listdir('D:/data/ext_VN_complete/')

result = []

for i in list_ext:
    path = 'D:/data/ext_VN_complete/' + str(i)
    pred = prediction(path, model)
    print (pred)
    temp = []
    temp.append (i)
    temp.append (pred)
    result.append(temp)


header = ['ID', 'Pred']
with open ('C:/Users/binhy/Desktop/patients2.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(result)
    
def scores_softmax (result):
    TP, FN, FP, TN = 0,  0, 0 , 0
    for i in result:
        
        if i[0][0] == '5' and int(i[1][0]) > 0.5:
            TP += 1
        if i[0][0] == '6' and int(i[1][0]) > 0.5:
            FP += 1
        if i[0][0] == '5' and int(i[1][0]) < 0.5:
            FN += 1
        if i[0][0] == '6' and int(i[1][0]) < 0.5:
            TN += 1
    return TP, FN, FP, TN

def scores_sigmoid (result):
    TP, FN, FP, TN = 0,  0, 0 , 0
    for i in result:

        if i[0][0] == '5':
            if i[1][0] < 0.5:
                TP += 1
            else:
                FN += 1
        if i[0][0] == '6':
            if i[1][0] < 0.5:
                FP += 1
            else:
                TN +=1
    return TP, FN, FP, TN

TP, FN, FP, TN = scores_sigmoid (result)

acc = (TP +TN) / (TP + TN +FN + FP)
pre = TP/ (TP + FP)
sen = TP/ (TP + FN)
spe = TN / (TN + FP)
print (acc, pre, sen, spe)
