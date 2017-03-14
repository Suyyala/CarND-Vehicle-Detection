import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os

from image_process import *

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML

import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from image_features import *
from classify import *

# NOTE: the next import is only valid for scikit-learn version <= 0.17
# for scikit-learn >= 0.18 use:
# from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split



cars = glob.glob(os.path.join("vehicles", "SRI_Augmented", "*.png"))
notcars = glob.glob(os.path.join("non-vehicles", "SRI_Augmented", "*.png"))

print('dataset sizes: ',  len(cars), len(notcars))

cv2.namedWindow('image' ,cv2.WINDOW_NORMAL)
cv2.resizeWindow('image', 800,800)

for car in cars:
    print(car)
    # Load an color image in grayscale 
    img = cv2.imread(car, cv2.IMREAD_COLOR)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    is_car = input('Is this a Car ? yes or no: ')
    if is_car == 'no':
        os.rename(car,'non-' + car)
        
   

cv2.destroyAllWindows()