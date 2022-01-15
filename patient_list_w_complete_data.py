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
    unique_kidney =[]
    for ID in IDs:
        files = os.listdir(directory + ID)
        only1 = False
        mask_basename = ''
        for file in files:
            f_name, file_ext = os.path.splitext (file)
            base = os.path.basename(f_name)
            if file_ext == '.nrrd':
                name  = mask_basename + base
        if 'l' and 'r' not in name:
            only1 = True
        if only1 == True:
            unique_kidney.append(ID)
    return unique_kidney        
        
with open ('C:/Users/binhy/Desktop/patients.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)
