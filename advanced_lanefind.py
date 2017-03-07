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



# Define a class to receive the characteristics of each line detection
class LaneLine():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        #polynomial coefficients for the most recent fit
        self.fit_history = [np.array([False])] 
        #radius of curvature of the line in some units
        self.curvature = []


def calculate_car_position(warped_img, lanes_info):

    if lanes_info is None:
        return None, None

    binary_warped = np.copy(warped_img)

    #retrieve cached values of fit values in pixel space
    left_lane, right_lane = lanes_info[:]
    current_fit = len(left_lane.fit_history)-1
    left_fit = left_lane.fit_history[current_fit]
    right_fit = right_lane.fit_history[current_fit]

    #car position which is bottom of the image close to y-max
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y_eval = binary_warped.shape[0] - 1
    left_fitx = left_fit[0]*y_eval**2 + left_fit[1]*y_eval + left_fit[2]
    right_fitx = right_fit[0]*y_eval**2 + right_fit[1]*y_eval + right_fit[2]
    
    lane_mid_point = (left_fitx + right_fitx) / 2

    offset_position = np.abs(np.ceil(binary_warped.shape[1]/2) - lane_mid_point)
    offset_position = offset_position * xm_per_pix
    return offset_position


# Example values: 1926.74 1908.48
def calculate_curvature(warped_img, lanes_info, margin=100):
   
    if lanes_info is None:
        return None, None

    #retrieve cached values of fit values in pixel space

    left_lane, right_lane = lanes_info[:]
    current_fit = len(left_lane.fit_history)-1
    left_fit = left_lane.fit_history[current_fit]
    right_fit = right_lane.fit_history[current_fit]

    #re-calculate co-efficents for real-world space

    binary_warped = np.copy(warped_img)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

     # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    y_eval = np.max(ploty)
    y_eval = binary_warped.shape[0]
    #print(y_eval)
    
    # Fit a second order polynomial to each
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)

    # Calculate the new radii of curvature in meters
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    #print(offset_position)

    #update latest curvature info
    left_lane.curvature.append(left_curverad)
    right_lane.curvature.append(right_curverad)
    if(len(left_lane.curvature) > 5):
        left_lane.curvature.pop(0)
    if(len(right_lane.curvature) > 5):
        right_lane.curvature.pop(0)

    return (left_curverad, right_curverad)


def fit_lines_optimize_sliding_window(warped_img, lanes_info, margin=100):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    if lanes_info is None:
        return None, None

    left_lane, right_lane = lanes_info[:]
    current_fit = len(left_lane.fit_history)-1
    left_fit = left_lane.fit_history[current_fit]
    right_fit = right_lane.fit_history[current_fit]

    binary_warped = np.copy(warped_img)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img2 = np.dstack((binary_warped, binary_warped, binary_warped))*255

    out_img2[lefty, leftx] = [255, 0, 0]
    out_img2[righty, rightx] = [0, 0, 255]


    #update latest left, right fit info
    left_lane.detected = True
    left_lane.fit_history.append(left_fit)
    right_lane.detected = True
    right_lane.fit_history.append(right_fit)
    if(len(left_lane.fit_history) > 5):
        left_lane.fit_history.pop(0)
    if(len(right_lane.fit_history) > 5):
        right_lane.fit_history.pop(0)

    return left_fitx, right_fitx, out_img2

def fit_lines_sliding_window(warped_img, lanes_info):
    binary_warped = np.copy(warped_img)
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[np.int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    out_img2 = np.dstack((binary_warped, binary_warped, binary_warped))*255

    out_img2[lefty, leftx] = [255, 0, 0]
    out_img2[righty, rightx] = [0, 0, 255]

    #Cache laneinfo and update latest values
    if lanes_info:
        #update latest left, right fit info
        left_lane, right_lane = lanes_info[:]
        left_lane.detected = True
        left_lane.fit_history.append(left_fit)
        right_lane.detected = True
        right_lane.fit_history.append(right_fit)
        if(len(left_lane.fit_history) > 5):
            left_lane.pop(0)
        if(len(right_lane.fit_history) > 5):
            right_lane.pop(0)

    return left_fitx, right_fitx, out_img2



def project_lines(original_img, warped_img, left_fitx, right_fitx, Minv, cam_dist):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    ploty = np.linspace(0, warped_img.shape[0]-1, warped_img.shape[0] )

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (color_warp.shape[1], color_warp.shape[0])) 
    # Combine the result with the original image
    #return newwarp
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    #plt.imshow(result)
    return result

# Edit this function to create your own pipeline.
def lanedetect_pipeline(img, cam_mat, cam_dist, vertices=None, lanes_info=None, debug=False):
    img = np.copy(img)

    #1. image  undistort
    img = cv2.undistort(img, cam_mtx, cam_dist, None, cam_mtx)
    if(debug is True):
         mpimg.imsave('./output_images/' +  'debug_undistort.jpg', img)

    #2. Apply Color, Gradient thresholds
    gradx_binary = abs_sobel_threshold(img, orient='x', threshold=(60,180))
    grady_binary = abs_sobel_threshold(img, orient='y', threshold=(60,180))
    mag_binary = mag_sobel_threshold(img, sobel_kernel=15, threshold=(60,150))
    dir_binary = dir_sobel_threshold(img, sobel_kernel=15, threshold=(0.7, 1.2))
    color_binary = hls_color_threshold(img, threshold=(170,255))

    combined_color_binary = np.zeros_like(gradx_binary)
    combined_color_binary[ ( (gradx_binary == 1) & (grady_binary == 1)) | (color_binary == 1)  | ((mag_binary == 1) & (dir_binary==1))] = 1
 
    if(debug is True):
         mpimg.imsave('./output_images/' +  'debug_combined_binary.jpg', combined_color_binary)


    #3. Perspecitve transform
    # apply the four point tranform to obtain a "birds eye view" of
    # the image
    warped_img, M, Minv = perspective_unwarp(combined_color_binary, vertices)
    if(debug is True):
         mpimg.imsave('./output_images/' +  'debug_perspective_unwarp.jpg', warped_img)
    
    #4. Lane detection using Sliding window
    lf, rf = None, None
    linefit_img = None
    lcd, rcd = None, None
    car_position_offset = 0.
    if lanes_info:
        left_lane, right_lane = lanes_info[:]
        if(left_lane.detected == True and right_lane.detected == True):
            lf, rf, linefit_img = fit_lines_optimize_sliding_window(warped_img, lanes_info)
        else:
            lf, rf, linefit_img = fit_lines_sliding_window(warped_img, lanes_info)
       
        #5. Calculate Radius of Curvature and Car position
        lcd, rcd = calculate_curvature(warped_img, lanes_info)
        car_position_offset = calculate_car_position(warped_img,lanes_info)

    else:
        lf, rf, linefit_img = fit_lines_sliding_window(warped_img, lanes_info)

    if debug is True:
         mpimg.imsave('./output_images/' +  'debug_linefit.jpg', linefit_img)

    #6. Project overlay back onto original image
    result = project_lines(img, warped_img, lf, rf, Minv, cam_dist)

    # overlay text resuluts
    if lcd is not None or rcd is not None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        text =  'curvature: ' + str(lcd) + ', '  + str(rcd)
        cv2.putText(result, text ,(50,50), font, 1,(255,255,255),2)
        text = 'car position offset: ' + str(car_position_offset)
        cv2.putText(result, text ,(50,100), font, 1,(255,255,255),2)

    return result


#0. Camera Calibration 
cam_mtx, cam_dist = calibrate_camera(debug=False)
print('camera matrix: ', cam_mtx)
print('distortion matrix: ', cam_dist)
straight_images = glob.glob('./test_images/straight*.jpg')
vertices = None
lanes_info = [LaneLine(), LaneLine()]
for fname in straight_images:
    print(fname)
    img = mpimg.imread(fname)
    #1. image  undistort
    img = cv2.undistort(img, cam_mtx, cam_dist, None, cam_mtx)
    mpimg.imsave('./output_images/' +  'undistort_' + fname.split('/')[-1] , img)
    #lines_img = perspective_detect_src_points(img)
    
    #vertices = np.array([[601,443],[680,443],[1020,657],[295,657]])
    #vertices = np.array([[552,474],[733,474],[996,640],[310,640]])
    #vertices = np.array([[552,474],[733,474],[1052,674],[267,678]])
    #vertices = np.array([[594,449],[687,449],[1052,674],[267,678]])
    #vertices = np.array([[572,461],[709,459],[1052,674],[267,678]])
    #vertices = np.array([[600,444],[680,441],[1059,681],[260,682]])
    vertices = np.array([[585,453],[696,453],[1059,681],[260,682]])
    
    warped_img, M, Minv = perspective_unwarp(img, vertices)
    mpimg.imsave('./output_images/' +  fname.split('/')[-1] , warped_img)


images = glob.glob('./test_images/test*.jpg')
for fname in images:
    print(fname)
    image = mpimg.imread(fname)
    result_img = lanedetect_pipeline(image, cam_mtx, cam_dist, vertices, lanes_info=None, debug=True)
    print(result_img.shape)
    #plt.imshow(result_img)
    mpimg.imsave('./output_images/' +  fname.split('/')[-1] , result_img)

#result = pipeline(image, cam_mtx, cam_dist)
def process_image(image):
    # NOTE: The output you return should be a color image (3 channel) for processing video below
    # TODO: put your pipeline here,
    # you should return the final output (image with lines are drawn on lanes)
    result  = lanedetect_pipeline(image, cam_mtx, cam_dist, vertices, lanes_info)
    return result

white_output = 'project_video_out.mp4'
clip1 = VideoFileClip("project_video.mp4", audio=False)
white_clip = clip1.fl_image(process_image) #NOTE: this function expects color images!!
white_clip.write_videofile(white_output, audio=False)





