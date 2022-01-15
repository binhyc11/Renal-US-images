import csv,os
with open ('C:/Users/binhy/Desktop/renalUS/Reference.csv') as file:
    csvreader = csv.reader (file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append (row)
header [0] = 'ID'
data_list = os.listdir('C:/Users/binhy/Desktop/renalUS/data')
data = []
for i in range (len(rows)):
    if rows[i][1] in data_list:
        data.append (rows[i])

def unique ():
    directory = 'D:/renalUS/data/'
    IDs = os.listdir(directory)
    uni =[]

    for ID in IDs:
        files = os.listdir(directory + ID)
        only1 = False
        names = ''
        for file in files:
            f_name, file_ext = os.path.splitext (file)
            base = os.path.basename(f_name)
            if file_ext == '.nrrd':
                names = names + base
        if ('l' and 'r' not in names) == True:
            only1 = True
        if only1 == True:
            uni.append(ID)
    return uni        
        
with open ('C:/Users/binhy/Desktop/patients.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
