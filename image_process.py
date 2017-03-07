import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os


def calibrate_camera(debug=False):
    
    objpoints = []
    imgpoints = []
    gray_image = None
    mtx = None
    dist = None
    nx = 9 # the number of inside corners in x
    ny = 6 # the number of inside corners in y
    
    #define object points (0,0,0), (1,0,0), (2,0,0)..(8,5,0) 
    objp = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    images = glob.glob('./camera_cal/calibration*.jpg')
    for fname in images:
        #print(fname)
        image = mpimg.imread(fname)

        # 2) Convert to grayscale
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        # 3) Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_image, (nx,ny), None)
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

    if len(imgpoints) > 0:
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray_image.shape[::-1] ,None,None)
    
    if debug is True:
        for fname in images:
            #print(fname)
            image = mpimg.imread(fname)
            image_undist = cv2.undistort(image, mtx, dist, None, mtx)
            mpimg.imsave('./output_images/' +  fname.split('/')[-1] , image_undist)
    return mtx, dist


def abs_sobel_threshold(img, orient='x', threshold=(0, 255)):
    #conver to grayscale
    img = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobel = None
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    if orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= threshold[0]) & (scaled_sobel <= threshold[1])] = 1
    return binary_output
  
def mag_sobel_threshold(img, sobel_kernel=3, threshold=(0, 255)):
    # Convert to grayscale
    img = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255 
    gradmag = (gradmag/scale_factor).astype(np.uint8) 
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= threshold[0]) & (gradmag <= threshold[1])] = 1

    # Return the binary image
    return binary_output


# Define a function to threshold an image for a given range and Sobel kernel
def dir_sobel_threshold(img, sobel_kernel=3, threshold=(0, np.pi/2)):
    # Grayscale
    img = np.copy(img)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction, 
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output =  np.zeros_like(absgraddir)
    binary_output[(absgraddir >= threshold[0]) & (absgraddir <= threshold[1])] = 1

    # Return the binary image
    return binary_output 

def hls_color_threshold(img, threshold=(0, 255)):
    #convert image to hls space
    img = np.copy(img)
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= threshold[0]) & (s_channel <= threshold[1])] = 1
    return binary_output

def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    NOTE: this is the function you might want to use as a starting point once you want to 
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).  
    
    Think about things like separating line segments by their 
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of 
    the lines and extrapolate to the top and bottom of the lane.
    
    This function draws `lines` with `color` and `thickness`.    
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    from random import randint
    colors = [[255, 0, 0], [0,255,0], [0,0,255]]
    for line in lines:
        for x1,y1,x2,y2 in line:
            color = colors[randint(0,2)]
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
        
    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    print(lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img

# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.
    
    `initial_img` should be the image before any processing.
    
    The result image is computed as follows:
    
    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

def region_of_interest(img, vertices):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def perspective_detect_src_points(img):
    img = np.copy(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Define a kernel size and apply Gaussian smoothing
    kernel_size = 5
    img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    # Define our parameters for Canny and apply
    low_threshold = 50
    high_threshold = 150
    edges = cv2.Canny(img, low_threshold, high_threshold)

    # Next we'll create a masked edges image using cv2.fillPoly()
    # This time we are defining a four sided polygon to mask
    imshape = img.shape
    vertices = np.array([[(100,imshape[0]-55),(imshape[1] * 0.5, imshape[0]* 0.595), (imshape[1] * 0.56, imshape[0] * 0.595), (imshape[1]-100,imshape[0]-55)]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    #low_threshold = 50
    #high_threshold = 150
    #edges = cv2.Canny(masked_edges, low_threshold, high_threshold)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on
    rho = 1 # distance resolution in pixels of the Hough grid
    theta = np.pi/180 # angular resolution in radians of the Hough grid
    threshold = 40     # minimum number of votes (intersections in Hough grid cell)
    min_line_len = 3 #minimum number of pixels making up a line
    max_line_gap = 150    # maximum gap in pixels between connectable line segments

    #lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    #print(lines)
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_len, max_line_gap)

    
    # Create a "color" binary image to combine with line image
    color_edges = np.dstack((edges, edges, edges)) 

    # Draw the lines on the edge image
    lines_edges = weighted_img(line_image, color_edges)
    return lines_edges



def perspective_unwarp(img, vertices=None):
    # 2) Convert to grayscale
    img = np.copy(img)
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    x1, y1 = vertices[0]
    x2, y2 = vertices[1]
    #cv2.line(img, (x1, y1), (x2, y2), color=[255,0.0], thickness=1)
    x1, y1 = vertices[1]
    x2, y2 = vertices[2]
    #cv2.line(img, (x1, y1), (x2, y2), color=[255,0.0], thickness=1)
    x1, y1 = vertices[3]
    x2, y2 = vertices[0]
    #cv2.line(img, (x1, y1), (x2, y2), color=[255,0.0], thickness=1)
    #x1, y1 = vertices[3]
    #x2, y2 = vertices[1]
    #cv2.line(img, (x1, y1), (x2, y2), color=[255,0.0], thickness=2)

    h_offset = 0 # offset for dst points
    w_offset = 200
    #print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    #h, w, c = img.shape

    src = np.float32(vertices)
    dst = np.float32([[w_offset, h_offset], [w-w_offset, h_offset], [w-w_offset, h-h_offset], [w_offset, h-h_offset]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (w,h), flags=cv2.INTER_LINEAR)
    return warped, M, Minv