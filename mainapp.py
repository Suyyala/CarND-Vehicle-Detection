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
from skimage import color
from image_features import *
from classify import *
from scipy.ndimage.measurements import label
from collections import deque



import sys, getopt

import random, string

def random_string(length):
   return ''.join(random.choice(string.ascii_lowercase) for i in range(length))
    

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (255,0,0), 4)
    # Return the image
    return img

# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


### TODO: Tweak these parameters and see how the results change.
color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 19  # HOG orientations
pix_per_cell = 16 # HOG pixels per cell
cell_per_block = 4 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [360, None] # Min and max in y to search in slide_window()


print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')

# Check the prediction time for a single sample
t=time.time()

# load the model from disk

svc = pickle.load(open('svc_model.p', 'rb'))
scaler = pickle.load(open('scaler.p', 'rb'))


class heat_windows:
    def __init__(self, max_len):
        self.win_q = deque(maxlen=max_len)
    def update(self, windows):
        self.win_q.append(windows)
    def get(self):
        win = []
        for e in  list(self.win_q):
            win = win + e
        return win

heatwindows = heat_windows(5)

#result = pipeline(image, cam_mtx, cam_dist)
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # you should return the final output (image with lines are drawn on lanes)
    #result  = lanedetect_pipeline(image, cam_mtx, cam_dist, vertices, lanes_info)
    #image = mpimg.imread(fname)
    draw_image = np.copy(image)
    # Uncomment the following line if you extracted training
    # data from .png images (scaled 0 to 1 by mpimg) and the
    # image you are searching is a .jpg (scaled 0 to 255)
    #image = image.astype(np.float32)/255
    windows_all = []
    hot_windows_all = []
    #window_sizes = [(96,96), (64,64)]
    window_sizes = [(128,128), (96, 96), (64, 64)]
    x_start_stops = [[400, None], [400, None], [400, None]]
    y_start_stops = [[400, None], [400, 620], [400,528]]
    window_index = 0
    for window_size in window_sizes:
        windows = slide_window(image, x_start_stop=x_start_stops[window_index],
                    y_start_stop=y_start_stops[window_index],
                    xy_window=window_size, xy_overlap=(0.7, 0.7))

        hot_windows = search_windows(image, windows, svc, scaler, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)
        hot_windows_all = hot_windows_all + hot_windows
        windows_all = windows_all + windows
        windows_index = window_index + 1
    #for window in hot_windows_all:
        #3) Extract the test window from original image
    #    test_img = cv2.resize(draw_image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
    #    mpimg.imsave(os.path.join('vehicles', random_string(10) + '.png'), test_img, format='png')
    
    heatwindows.update(hot_windows_all)
    heatwin_list = heatwindows.get()
    heat = np.zeros_like(draw_image[:,:,0]).astype(np.float)
    heat = add_heat(heat, heatwin_list)
    heat = apply_threshold(heat, 20)
    heatmap = np.clip(heat, 0, 255)
    labels = label(heatmap)
    
    result = draw_labeled_bboxes(draw_image, labels)
    #result = draw_boxes(result, hot_windows, color=(0, 0, 255), thick=3)
    #result = draw_boxes(result, windows1, color=(0, 255, 0), thick=1)
    #result = draw_boxes(result, windows2, color=(0, 255, 255), thick=1)
    
    return result

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"hi:",["ifile="])
    except getopt.GetoptError:
        print('mainapp.py -i <inputfile>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('mainapp.py -i <inputfile>')
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg

    print('Input file is "', inputfile)
    video_input = inputfile

    clip = VideoFileClip(video_input, audio=False)
    clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
    #clip.preview()
    clip.write_videofile(video_input.split('.')[0] + '_out' + '.mp4', audio=False)

if __name__ == "__main__":
   main(sys.argv[1:])