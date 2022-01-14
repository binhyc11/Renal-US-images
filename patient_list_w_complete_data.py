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

with open ('C:/Users/binhy/Desktop/patients.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(data)