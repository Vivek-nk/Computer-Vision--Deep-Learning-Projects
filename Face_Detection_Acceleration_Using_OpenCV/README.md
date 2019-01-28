# Face Detection Acceleration On Raspberry PI Using OpenCV

## Background

Object detection using Haar feature-based cascade classifiers is an effective face detection method proposed by Paul Viola and Michael Jones in their paper, “Rapid Object Detection using a Boosted Cascade of Simple Features” in 2001. The approach uses machine learning to construct a cascade function that is trained with many positive images (images of faces) and negative images (images without faces). Existing training files for front faces have already been generated and there are numerous publicly available trained object databases (pedestrian, car, sign, etc). An assignment later in the semester will include training your own individual classifier. For now, we will visit the step of extract features from an image and comparing them to a trained set. For this, some of the example Haar features shown in below image are used. Each feature filter (also called a kernel or a mask) generates a single value obtained by subtracting sum of pixels under white rectangle from sum of pixels under black rectangle.

All possible sizes and locations of each kernel are used to calculate the feature results. This tends to amount to a lot of computation. For example, for a 24x24 window, there are over 160000 features. For each feature calculation, the system must find the sum of pixels under white and black rectangles. To solve this efficiently, the integral images is exploited to simplify the calculation of sum of pixels, how large may be the number of pixels, to an operation involving just four pixels. 

Among the numerous features calculated, most of them are irrelevant. For example, consider the image below. Top row shows two good features. The first feature selected seems to focus on the property that the region of the eyes is often darker than the region of the nose and cheeks. The second feature selected relies on the property that the eyes are darker than the bridge of the nose. But the same windows applying on cheeks or any other place is irrelevant. So how do we select the best features out of 160000+ features? The process is achieved by Adaptive Boosting (Adaboost).

For AdaBoost, the system applies each and every feature on all the training images. For each feature, it finds the best threshold which will classify the faces to positive and negative. But obviously, there will be errors or misclassifications. The system select the features with minimum error rate, which means selecting the features that best classifies the face and non-face images. The process is not as simple as this. Each image is given an equal weight in the beginning. After each classification, weights of misclassified images are increased. Then again same process is done. New error rates are calculated and new weights. The process is continued until required accuracy or error rate is achieved or required number of features are found.

In an image, most of the image region is non-face region. So it is a better idea to have a simple method to check if a window is not a face region. If it is not, discard it in a single shot. Don’t process it again. Instead focus on region where there can be a face. This way, we can find more time to check a possible face region. For this Viola and Jones introduced the concept of Cascade of Classifiers. Instead of applying all the 6000 features on a window, group the features into different stages of classifiers and apply one-by-one. Normally first few stages will contain very less number of features. If a window fails the first stage, discard it. This frees the system from consider running remaining features on it. If it passes, apply the second stage of features and continue the process. The window that passes all stages is a face region. 

The original Viola and Jones detector had 6000+ features with 38 stages with 1, 10, 25, 25 and 50 features in first five stages. (Two features in the above image is actually obtained as the best two features from Adaboost). According to authors, on an average, 10 features out of 6000+ are evaluated per sub-window. This is a simple intuitive explanation of how Viola-Jones face detection works. Read paper for more details.

# Haar-cascade Detection in OpenCV

OpenCV comes with a trainer as well as detector. If you want to train your own classifier for any object like car, planes etc. you can use OpenCV to create one. Its full details are given here: Cascade Classifier Training. This assignment deals with detection not training. OpenCV already contains many pre-trained classifiers for face, eyes, smile etc. Those XML files are stored in opencv/data/haarcascades/ folder. Let’s create face and eye detector with OpenCV. First we need to load the required XML classifiers. Then load our input image (or video) in grayscale mode.

%pylab inline

import cv2

import numpy as np

import re


Next, if we were to read the face and perform color to grayscale conversion.

img = cv2.imread('image.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

To find the faces in the image, the cascade.detectMultiScale function will be used. If faces are found, it returns the positions of detected faces as Rect(x,y,w,h). Once we get these locations, we can create a Region of Interest (ROI) for the face and apply eye detection on this ROI (since eyes are always on the face.)

def detectFace(path, cascade):

    # Read image and convert to gray
    
    img = cv2.imread(path)
    
    img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
    
    # Base parameters 
    
    # Hint: re-write detectFace to pass in changes to parameters
    
    rects = cascade.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20),(200,200))
   
    if len(rects) == 0:
    
        return [], img
        
    rects[:, 2:] += rects[:, :2]
        
    #Return the location and  original image
    
    return rects, img
    

def readTrainingFile(trainingFile):

    try:
    
        file_in = open(trainingFile, 'r')   #check if file exists
        
    except IOError:
    
        print 'Can\'t open file ', trainingFile, ', exiting...'
        
        sys.exit()
        
    imageDictionary = {}
    
    for line in file_in:
    
        columns = line.split(":")
        
        imageFileName = str(columns[0])
        
        val = int(columns[1])
        
        imageDictionary[imageFileName] = val
        
        # uncomment next line if you want to view the full training file
        
        #print imageFileName,val
        
    return imageDictionary
    


Applying the Haar detector involves running the cascaded classifier with python function: cascade.detectMultiScale, after the cascade has been properly loaded from an .xml file.

You can find background material on the Python OpenCV functions:
http://docs.opencv.org/modules/objdetect/doc/cascade_classification.html

##Parameters:

  cascade – Haar classifier cascade that can be loaded from XML or YAML file using When the cascade is not needed anymore, release it using cvReleaseHaarClassifierCascade(&cascade).
  
	image – Matrix of the type CV_8U containing an image where objects are detected.
  
	scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
  
	minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it. The higher the number the fewer false positives you will get. If this parameter is set to something larger than 0, the algorithm will group intersecting rectangles and only return those that have overlapping rectangle greater than or equal to the minimum number of nearest neighbors. If this parameter is set to 0, all rectangles will be returned and no grouping will happen, which means the results may have intersecting rectangles for a single face.
  
	flags – Parameter type of Haar detection
  
	minSize – Minimum possible object size. Objects smaller than that are ignored.
  
	maxSize – Maximum possible object size. Objects larger than that are ignored.
  
  
Example use: 

rects = cascade.detectMultiScale(gray, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20), (200,200))

Image: gray, ScaleFactor: 1.3, MinNeighbors : 4,  MinNeighbors: 4, flags : cv2.cv.CV_HAAR_SCALE_IMAGE, minSize : (20x20), maxSize : (200x200)

## Overview

You will analyze the OpenCV function detectMultiScale and the database of face images with respect to automating a “sweep” of parameter values.  A sweep is an experimental study in which parameters are modified to determine the most favorable setting of results (accuracy, execution time, power, etc).

## Base evaluation:

Image: gray, ScaleFactor: 1.3, MinNeighbors : 4,  MinNeighbors: 4, flags : cv2.cv.CV_HAAR_SCALE_IMAGE, minSize : (20x20), maxSize : (200x200)

Parameter	Base Value	Sweep

ScaleFactor	1.3	1.3,1.6, 2.0, 3.0, 4.0

MinNeighbors	4	4,8,16,32

Flags	cv2.cv.CV_HAAR_SCALE_IMAGE	No sweep

minSize	(20,20)	(20,20),(40,40), (80,80), (160,160)

maxSize	(200,200)	(200,200),(150,150),(100,100),(50,50)


# In this project: 

Each parameter has 4 distinct sweep settings.Sweep 1 parameter and keep ALL other parameters at their base values.

[Task] Execution Time- Modify the provided so that every experimental sweep tracks the execution time of the face database [A]. Compare the execution time should show 4 graphs of your choice of format, but that a respective graph is provided for each sweep of ScaleFactor, MinNeighbors, minSize, and maxSize.  

[Task] Accuracy – Compare the accuracy of each parameter change run for face database [A]. 
Compare the face detection accuracy by showing 4 graphs of your choice of format, but that a respective graph is provided for each sweep of ScaleFactor, MinNeighbors, minSize, and maxSize.  

[Task] Execution Time versus Accuracy - Plot 4 execution time versus accuracy plots for each of the experiments: ScaleFactor, MinNeighbors, minSize, and maxSize. Perform this graphing on training set A.

[Task] Comparison Across Training Sets [A,B,C,D,E] – Evaluate the accuracy changes across input sets. Specifically you should show some results that categorize which of the input sets is most susceptible to changes in: ScaleFactor, MinNeighbors, minSize, and maxSize.


