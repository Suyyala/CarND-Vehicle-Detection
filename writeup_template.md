
The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, can apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run  pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car.png
[image2]: ./output_images/notcar.png
[image3]: ./output_images/car_hog_ch0.png
[image4]: ./output_images/notcar_hog_ch0.png
[image5]: ./output_images/training_result.png
[image6]: ./output_images/test1_windows.jpeg
[image7]: ./output_images/test1_hotwindows.jpeg
[image8]: ./output_images/test1_heatmap.jpeg
[video1]: ./project_video.mp4


####1. Extracted  features (HOG + Histogram + ) from the training images.

The code for this step is contained in files 'train_model.py' from lines #31 to #83 and in  'image_features.py'.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=19`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image3]
![alt text][image4]

####2. Image feature parameters.

I have tried various parameters and tested the classifer  on test_images. For me, 'YUV' color space with above HOG parameters given
very good results compared to RGB space.

####3. Training the model

The code for this step is contained in file 'train_model.py' from lines #86 to #163.

I have used LinearSVM Classifer to train the model with image feature vectors extracted from Step#1. After shuffling the input data (Udacity provided data sets) I have split the data into training and test with 80, 20  ratio. Training data has been normalized using StandardScalar() before fit into LinearSVM classifer.

After the model is trained, tested accuracy on test data, which yeild score close to .96 to .97. There were many false positives found with trained model, when made me  experiment with various combinations of image feature parameters described in step #1. And also, augmented vehicle data from udacity annotated dataset and some of my own data set to improve the false positives, which I beleve improved classifer prediction.

https://github.com/udacity/self-driving-car/tree/master/annotations

After, I am satified with test accuracy,saved LinearSVM model for use in my main application pipeline to detect vehicles in project video.

![alt text][image5]

###Sliding Window Search

####1. Implementation

With training model saved, rest of the  vehicle detection pipeline is implemeted in mainapp.py from lines #102 to #151.

The code for sliding window search implemented is in file 'classify.py' from lines #38 to #74, which generates list of windows
given the window size and overlapping ratio. I have used this implemetation to generate sliding windows with sizes [ (128,128), (96,96) and (64, 64). This function also takes parameters to reduce the image search region.

Now SVM classifier is fed with Image freatures extracted from Sliding windows created above to classify for cars. If Classifier predictions to be a Car, then the windows are added to list of hot_windows as potential candiates. 

The code for Image extractio and Slding windows prediction is implemented in file 'classify.py' from lines #81 to #157.

After aggregating list of hot_windows, a heatmap implementation is used to filter out the false positives and as well find bounding boxes with overlapping regions. I have used heatmap implementation with aggregated over last 5 frames with thresholds ranging from 
10 to 15. 

The code for heatmap implemetation can be found between in file mainapp.py between lines #138 to #144. The helper functions are 
implemented in Classify.py (lines #18 to #32)

![alt text][image6]

####2. Performance

Ultimately I searched on three scales using  HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result. I have reduced image search regions for different window sizes to speed up the pipeline. Here are some example images:

![alt text][image7]
---

### Video Implementation

####1.
My results video can be found in gitrepo (./project_video_out.mp4)

####2.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  
Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]


###Discussion

I have spent lot of time fighting false positives, which made to collect more data of trees, roads, traffic signs, dividers etc to reduce the false positives.  However, I have quickly realized it is balanced boat that requires lot of positive instead of lot of negative examples(which will never be exhaustive). Althogh, this feature based implemented works fine to an extent howevet, it is not robust enuogh to changing environment requiring lot of features. Also, need higher framerates and lot of processing power to do realtime 
detection using this approach.

I think deep learning or neural network models along with image feature vectors makes the pipeline more robust and may be realtime implementation with 60 to 90 fps image frame rates. 

