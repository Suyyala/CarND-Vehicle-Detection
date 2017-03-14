
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



clip = VideoFileClip("project_video.mp4", audio=False)
#clip = clip.fl_image(process_image) #NOTE: this function expects color images!!
clip_len = 10
for i in range(0,51):
    print(i*clip_len, (i+1)*clip_len)
    if (i+1) * clip_len > 51:
        break
    clip1 = clip.subclip( i * clip_len,  (i+1) * clip_len)
    #clip.preview()
    clip1.write_videofile('project_' + str(clip_len) + '_' + str(i) + '.mp4', audio=False)

