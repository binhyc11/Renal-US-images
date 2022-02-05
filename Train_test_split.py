import os, shutil
import numpy as np
from sklearn.model_selection import train_test_split

u55 = os.listdir('D:/renalUS/exp_U55')
b65 = os.listdir('D:/renalUS/exp_B65')

data =[]
label = []

for i in u55:
    data.append('D:/renalUS/exp_U55/' + i)
    label.append(5)

for i in b65:
    data.append('D:/renalUS/exp_B65/' + i)
    label.append(6)

X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.2)

for i in range (len(X_train)):
    basename = os.path.splitext(os.path.basename (X_train[i]))[0]
    basename = os.path.splitext (basename)[0]
    shutil.copyfile(X_train[i], 'D:/train/' + str(y_train[i]) + '_' + basename + '.png')
    
for i in range (len(X_test)):
    basename = os.path.splitext(os.path.basename (X_test[i]))[0]
    basename = os.path.splitext (basename)[0]
    shutil.copyfile(X_test[i], 'D:/test/' + str(y_test[i]) + '_' + basename + '.png')
