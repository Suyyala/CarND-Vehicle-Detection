import csv

with open('labels.csv') as fin, open('labels_conv.csv', 'w') as fout:
    o=csv.writer(fout)
    for line in fin:
        o.writerow(line.split())
