import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os


import time


import pandas as pd
import numpy as np


"""
df = pd.read_csv(os.path.join('object-detection-crowdai', 'labels.csv'))
cars = df[df['Label'] == 'Car']
count = 0
for index, car in cars.iterrows():
    print(car)
    print('processing: ' + car['Frame'])
    if car['ymin'] == car['ymax'] or car['xmin'] == car['xmax']:
        continue
    if car['xmax'] - car['xmin'] < 64:
        continue
    if car['ymax'] - car['ymin'] < 64:
        continue
    img = mpimg.imread(os.path.join('object-detection-crowdai', car['Frame']) )
    print(img.shape)
    img = cv2.resize(img[car['ymin']:car['ymax'], car['xmin']:car['xmax']], (64, 64))
    mpimg.imsave(os.path.join('object-detection-crowdai', 'extract',  car['Frame'].split('.')[0] + '-' + str(count) + '.png'), img, format='png')

"""
df = pd.read_csv(os.path.join('object-dataset', 'labels_conv.csv'))
cars = df[df['Label'] == '"car"']
#print(df['Label'])
count = 0
for index, car in cars.iterrows():
    #print(car)
    #print('processing: ' + car['Frame'])
    if car['ymin'] == car['ymax'] or car['xmin'] == car['xmax']:
        continue
    if car['xmax'] - car['xmin'] < 64:
        continue
    if car['ymax'] - car['ymin'] < 64:
        continue
    img = mpimg.imread(os.path.join('object-dataset', car['Frame']) )
    #print(img.shape)
    img = cv2.resize(img[car['ymin']:car['ymax'], car['xmin']:car['xmax']], (64, 64))
    mpimg.imsave(os.path.join('object-dataset', 'extract',  car['Frame'].split('.')[0] + '-' + str(count) + '.png'), img, format='png')          
#cars = glob.glob(os.path.join("vehicles", "SRI_Augmented", "*.png"))
#notcars = glob.glob(os.path.join("non-vehicles", "SRI_Augmented", "*.png"))
#print('dataset sizes: ',  len(cars), len(notcars))
