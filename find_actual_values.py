import os, csv

with open ('C:/Users/binhy/Desktop/heather.csv') as file:
    csvreader = csv.reader (file)
    header = []
    header = next(csvreader)
    header.append ('actual1')
    header.append ('actual2')
    heather = []
    for row in csvreader:
        heather.append (row)

with open ('C:/Users/binhy/Desktop/patients_list.csv') as file:
    csvreader = csv.reader (file)
    head = []
    head = next(csvreader)
    actual = []
    for row in csvreader:
        actual.append (row)

for pts in heather:
    ID = pts[0][:8]
    for i in actual:
        if ID == i [1][:8]:
            pts.append (i[3])
            

with open ('C:/Users/binhy/Desktop/heather_actual.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(heather)
