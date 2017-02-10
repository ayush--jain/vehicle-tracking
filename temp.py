import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from moviepy.editor import VideoFileClip

from sklearn.model_selection import train_test_split

colorspace = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL'
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [None, None] # Min and max in y to search in slide_window()

# img = cv2.imread('test_images/test3.jpg')
# img = img[350: img.shape[0]-100, 600:]
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# plt.imshow(img)
# plt.show()




# feature_array, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
# 			    cells_per_block=(cell_per_block, cell_per_block), visualise=True, feature_vector=False)

def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
  
	features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
	               cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
	               visualise=vis, feature_vector=feature_vec)
	return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(image, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
	file_features = []
	# Read in each one by one
	# image = mpimg.imread(file)
	# image = (image*255.).astype(np.uint8)
	# apply color conversion if other than 'RGB'
	if color_space != 'RGB':
	    if color_space == 'HSV':
	        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
	    elif color_space == 'LUV':
	        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
	    elif color_space == 'HLS':
	        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
	    elif color_space == 'YUV':
	        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
	    elif color_space == 'YCrCb':
	        feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
	else: feature_image = np.copy(image)      

	# if spatial_feat == True:
	#     spatial_features = bin_spatial(feature_image, size=spatial_size)
	#     file_features.append(spatial_features)
	# if hist_feat == True:
	#     # Apply color_hist()
	#     hist_features = color_hist(feature_image, nbins=hist_bins)
	#     file_features.append(hist_features)
	if hog_feat == True:
	# Call get_hog_features() with vis=False, feature_vec=True
	    if hog_channel == 'ALL':
	        hog_features = []
	        for channel in range(feature_image.shape[2]):
	            hog_features.extend(get_hog_features(feature_image[:,:,channel], 
	                                orient, pix_per_cell, cell_per_block, 
	                                vis=False, feature_vec=True))      
	    else:
	        hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
	                    pix_per_cell, cell_per_block, vis=False, feature_vec=True)
	    # Append the new feature vector to the features list
	    file_features.append(hog_features)

    #9) Return concatenated array of features
	return np.concatenate(file_features)


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        image = (image*255.).astype(np.uint8)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      

        # if spatial_feat == True:
        #     spatial_features = bin_spatial(feature_image, size=spatial_size)
        #     file_features.append(spatial_features)
        # if hist_feat == True:
        #     # Apply color_hist()
        #     hist_features = color_hist(feature_image, nbins=hist_bins)
        #     file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                        orient, pix_per_cell, cell_per_block, 
                                        vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)        
            else:
                hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
#                         hist_bins=32, orient=9, 
#                         pix_per_cell=8, cell_per_block=2, hog_channel=0,
#                         spatial_feat=True, hist_feat=True, hog_feat=True):
#     # Create a list to append feature vectors to
# 	features = []
# 	for file in images:
# 		image = mpimg.imread(file)
# 		image = (image*255.).astype(np.uint8)

# 		file_features = single_img_features(image, color_space=color_space, 
#                             spatial_size=spatial_size, hist_bins=hist_bins, 
#                             orient=orient, pix_per_cell=pix_per_cell, 
#                             cell_per_block=cell_per_block, 
#                             hog_channel=hog_channel, spatial_feat=spatial_feat, 
#                             hist_feat=hist_feat, hog_feat=hog_feat)
# 		features.append(file_features)

# 	return features

images = glob.glob('vehicles/*/*.png')
cars = []
for image in images:
    cars.append(image)

images = glob.glob('non-vehicles/*/*.png')
notcars = []
for image in images:
    notcars.append(image)

import random
ran = random.randint(0, len(cars))
img = cv2.imread('vehicles/KITTI_extracted/17.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2 = cv2.imread('non-vehicles/Extras/extra6.png')
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

_, axs = plt.subplots(1, 2)
axs = axs.ravel()
plt.figure(figsize=(2,2));
axs[0].set_xticklabels([])
axs[0].set_yticklabels([])
axs[0].imshow(img)
axs[0].set_title('Car')

axs[1].set_xticklabels([])
axs[1].set_yticklabels([])
axs[1].imshow(img2)
axs[1].set_title('Not Car')
    # plt.title(label[0])
plt.show();


# print (len(cars))
# print (len(notcars))
