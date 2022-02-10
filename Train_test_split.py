import os, shutil, csv
from sklearn.model_selection import train_test_split

ID_55_65 = []
list_55_65 = os.listdir('D:/data/ext_in_range_55_65/')
for i in list_55_65:
    ID = i[:8]
    if ID not in ID_55_65:
        ID_55_65.append (ID)

with open ('C:/Users/binhy/Desktop/patients_list.csv') as file:
    csvreader = csv.reader (file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append (row)

path_0, path_1 = [], []
        
for i in rows:
    for j in list_55_65:
        path_j = 'D:/data/ext_in_range_55_65/' + j
        ID = j[:8]
        if ID == i[0]:
            if i[1] == '0':
                path_0.append(path_j)
            else:
                path_1.append(path_j)

test_5 = os.listdir('D:/data/test_exp/5/')
for i in test_5:
    path = 'D:/data/test_exp/5/' + i
    path_0.append (path)

train_5 = os.listdir('D:/data/train_exp/5/')
for i in train_5:
    path = 'D:/data/train_exp/5/' + i
    path_0.append (path)

test_6 = os.listdir('D:/data/test_exp/6/')
for i in test_6:
    path = 'D:/data/test_exp/6/' + i
    path_1.append (path)

train_6 = os.listdir('D:/data/train_exp/6/')
for i in train_6:
    path = 'D:/data/train_exp/6/' + i
    path_1.append (path)
    


train_0, test_0 = train_test_split(path_0, test_size=0.2)
train_1, test_1 = train_test_split(path_1, test_size=0.2)

for i in train_0:
    name = os.path.basename (i)
    shutil.copyfile(i, 'D:/train/0/' + name)

for i in test_0:
    name = os.path.basename (i)
    shutil.copyfile(i, 'D:/test/0/' + name)

for i in train_1:
    name = os.path.basename (i)
    shutil.copyfile(i, 'D:/train/1/' + name)

for i in test_1:
    name = os.path.basename (i)
    shutil.copyfile(i, 'D:/test/1/' + name)
