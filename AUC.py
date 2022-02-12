from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os, csv
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


X_test = []
y_test = []

list_ext = os.listdir('D:/data/test_exp_com/')

result = []

for i in list_ext:
    path = 'D:/data/test_exp_com/' + str(i)
    img = image.load_img(path, target_size=(195, 334))
    x = image.img_to_array(img)
    x/= 255.0
    X_test.append(x)
    y_test.append([1,0] if i[0]=='5' else [0,1])

X_test = np.array(X_test)    
y_test = np.array(y_test)
    
model = load_model("D:/models/sof_11x32_9x64.h5")
model.compile (loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])   
    
scores = model.evaluate(X_test, y_test, verbose = 1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
score = model.predict(X_test)
# print (score)

predcs1 = score[:, 0]
new_label = y_test[:, 0]
tn, fp, fn, tp = confusion_matrix(new_label, predcs1.round()).ravel()

sensitivity = tp / (tp+fn) # recall
specificity = tn / (tn+fp)
precision = tp / (tp+fp)
print('sensitivity or recall', sensitivity)
print('specificity', specificity)
print('precision', precision)
print('F1 score', 2*precision*sensitivity/(precision + sensitivity))

fpr, tpr, threshold = metrics.roc_curve(new_label, predcs1)
roc_auc = metrics.auc(fpr, tpr)
print("Area Under Curve is: %f" % (roc_auc))
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label='Model "sof_11x32_9x64" (area = {:.3f})'.format(roc_auc))

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()




# def prediction (path, model):    
#     img = image.load_img(path, target_size=(195, 334))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     X_test.append(x)
    
    
#     images = np.vstack([x])
#     classes = model.predict(images)
#     return (classes[0])

    


    # pred = prediction(path, model)
    # print (pred)
    # temp = []
    # temp.append (i)
    # temp.append (pred)
    # result.append(temp)


# header = ['ID', 'Pred']
# with open ('C:/Users/binhy/Desktop/patients.csv', 'w', encoding='UTF8', newline='') as f:
#     writer = csv.writer(f)
#     writer.writerow(header)
#     writer.writerows(result)
