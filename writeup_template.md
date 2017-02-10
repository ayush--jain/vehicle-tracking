##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png
[image2]: ./output_images/vehicle_hog.png
[image2_1]: ./output_images/non-vehicle_hog.png
[image3]: ./output_images/test_5_bb.png
[image4]: ./output_images/test5_final.png
[image4_1]: ./output_images/test1_final.png
[image4_2]: ./output_images/test3_final.png
[image5]: ./output_images/test5_hm.png
[image6]: ./examples/labels_map.png
[image7]: ./output_images/test5_final.png
[video1]: ./project_video.mp4
---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in lines 118 through 167 of the file called `initial.py`.

I started by reading in all the `vehicle` and `non-vehicle` images (lines 189 to 197).  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YUV` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

Vehicle image:
![alt text][image2]

None-Vehicle image:
![alt text][image2_1]


####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found the following values to give the best results:

* colorspace = 'YUV' 
* orient = 9
* pix_per_cell = 8
* cell_per_block = 2
* hog_channel = 'ALL'

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

This is implemented in lines 210-233 in the file `initial.py`
I trained a linear SVM using the extracted features from the training dataset described above. This was done after diving the features into training and testing sets to get the score. Also, I scaled and normalized the data before feeding it to the classifier.
The classifier would be trained to predict 1 if car was present or 0 if it wasn't. I predicted the test set and compared the predicted values with the true labels. I achieved an accuracy score of about 98%

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

This is implemented in lines 237-315 in `initial.py` where I have a function for sliding the window and searching for vehicles in this window.
I scanned the window for the lower half of the image. I cropped 350 px from the top, 100 px from the bottom. The sliding window overlaps for 0.85 (both x and y). I decided on these values through experimentation and found these to give the best output. The classifier checks if there is a car in the frame and all windows where a vehicle is predicted is returned. A sample image after this step is:

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I adjusted the extent of overlap on the sliding windows to 0.85 which stablized the results. I also tweaked the value of heatmap thresholds and cropped the images further(for running sliding windows) to reduce false detection.
Ultimately I searched the window using YUV channel HOG features which provided a nice result.  Here are some example images:

![alt text][image4]
![alt text][image4_1]
![alt text][image4_2]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_result.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions. I constructed bounding boxes to cover the area of each blob detected.  

The creation of heatmap and threshold can be found from lines 318-332 in `initial.py` and the code for making a bounding box from the resulting heatmap can be found from lines 342 through 360 in the same file.

Here's an example result showing the heatmap from a frame of video and the bounding boxes then overlaid on the last frame of video:

### Here is a frame and its corresponding heatmaps:

![alt text][image5]

### Here the resulting bounding boxes are drawn onto the same frame:
![alt text][image7]

More examples images can be found in the folder `output/images`.



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had several issues with the project, especially with tuning the parameters. With some parameters, the false positives were high and with some, the car wasn't being detected all the time. After several hours, I think I have a reasonable balance where the car is detected all the time with minimal false positives. It is not perfect but in the time I had for the project, this is the best I could do.

I also could not incorporate the color transform because of an issue with the array sizes and found I got decent results with its exclusion as well.

The pipeline is likely to fail when there are overalpping vehicles or noise in the environment. I think deep learning approaches would be more robust and can generalize well and get better results than SVM and its tuning.
