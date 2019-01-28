#  code courtesy of:    Dr. Dan Connors
#  edited by:           Todd Fulton       on: 12/4/2017
#  Purpose:             Final Project HWSW interface
#  
#
#%pylab inline *************for plotting I have matplotlib*******
import matplotlib.pyplot as plt # ****instead of %pylab inline***
import os
#import cv
import cv2
import numpy as np
import re
import pylab as pl
from fnmatch import fnmatch
#Original: call def detectFace(path, cascade):

# new call : now passes in four new parameters : scaleFactor, minNeighbors, minSize, maxSize
def detectFace(path, cascade, scaleFactor=1.3, minNeighbors=4, minSize=(20,20), maxSize=(200,200)):   
 
    # Read image and convert to gray
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
     
    # Base parameters 
    # Hint: re-write detectFace to pass in changes to parameters
    # Original rects = cascade.detectMultiScale(img, 1.3, 4, cv2.cv.CV_HAAR_SCALE_IMAGE, (20,20),(200,200))
    rects = cascade.detectMultiScale(img, scaleFactor, minNeighbors,cv2.CASCADE_SCALE_IMAGE,minSize, maxSize)

    if len(rects) == 0:
        return [], img
    rects[:, 2:] += rects[:, :2]
    
    #Return the location and  original image
    return rects, img

def ComputeFaceAccuracy(faceMatches, faceGroup, faceTable):
    correct = 0
    for i in range(len(faceGroup)):
        faceimagePath = faceGroup[i]
        faceimageFile = faceimagePath.split('/')[-1]
        foundFaces = faceMatches[i]
        refFaces = faceTable[faceimageFile]
        
        # Check that the faces are correct
        if (refFaces == foundFaces):
            correct = correct + 1
    return float(correct)/float(len(faceGroup))

def readTrainingFile(trainingFile):
    try:
        file_in = open(trainingFile, 'r')   #check if file exists
    except IOError:
        print ('Can\'t open file ', trainingFile, ', exiting...')
        sys.exit()
    
    imageDictionary = {}
    for line in file_in:
        columns = line.split(":")
        imageFileName = str(columns[0])
        val = int(columns[1])
        imageDictionary[imageFileName] = val
        #uncomment next line if you want to view the full training file
        #print imageFileName,val

    return imageDictionary
# Sweep parameter lists
scale_factor = [ 1.3, 1.6, 2.0, 3.0, 4.0] # doesnt work for 1.0
min_neighbors = [4, 8, 16, 32]
min_size = [(20,20), (40,40), (80,80), (160,160)]
max_size = [(200,200), (150,150), (100,100), (50,50)]

faceDataLocation = '/home/facedata/2003/01/13/big'

# Location of face images : running all
faceDirList = ["FaceData/A", "FaceData/B","FaceData/C","FaceData/D","FaceData/E"]
faceGroupList = ["A","B","C","D","E"]
groupList = ["A","B","C","D","E"]

# Only do one directory at a time faces, group "A"
faceDirList = ["FaceData/A"]
groupList = ["A"]
faceGroupList = ["A"]


# Location of AdaBoost Haar Cascade
OPENCV_PATH = "/home/pi/opencv-3.1.0/data" #*******You MUST have the correct file path here ******
HAAR_CASCADE_PATH = OPENCV_PATH + "/haarcascades"
face_cascade = HAAR_CASCADE_PATH + '/haarcascade_frontalface_default.xml'
cascade = cv2.CascadeClassifier(face_cascade)

# Define filename to match : any .jpg 
imageFilePattern = "*.jpg"
    
# Create an empty list: will hold OpenCV images
imageList = list()   

# Get the known results of number of faces
faceTable = readTrainingFile("FaceData/training.txt")
# faceGroup
faceGroup = {}
for iface_dir in np.arange(len(faceDirList)):
    group = faceGroupList[iface_dir]
    faceGroup[group] = []
    
    for path, subdirs, files in os.walk(faceDirList[iface_dir]):
        for name in files:
            if fnmatch(name, imageFilePattern): # check "*.jpg"
                # Create file name
                faceFile = os.path.join(path, name)
                faceGroup[group].append(faceFile)
# Use 4 lists for execution time: one for each parameter being swept
SweepTimeMinSize = []
SweepTimeMaxSize = []
SweepTimeScale = []
SweepTimeNeighbors = []

# Use 4 lists for accuracy: one for each parameter being swept
SweepAccuracyMinSize = []
SweepAccuracyMaxSize = []
SweepAccuracyScale = []
SweepAccuracyNeighbors = []

# one group, "A"
groupList=["A"] 
for group in groupList:
    
   print ("Working on group", group)   
   # Sweep one parameter at a time
    
   # Sweep the "scale factor" variable
   print ("Sweep scale factor start.") 
   for item in scale_factor:
      faceMatches = []
        
      # Start timer
      t_start = cv2.getTickCount()
      for index in range(len(faceGroup[group])):
          rects, img = detectFace(faceGroup[group][index], cascade, scaleFactor=item)
          faceMatches.append(len(rects))
      
      # Stop timer
      t_stop = cv2.getTickCount()
      t_total = (t_stop - t_start) / cv2.getTickFrequency()
            
      # Append time to track
      SweepTimeScale.append(t_total)
            
      # Compare accuracy
      accuracy = ComputeFaceAccuracy(faceMatches, faceGroup[group], faceTable)
      SweepAccuracyScale.append(accuracy)
   print ("Sweep scale factor complete.")
            
   # Sweep the "neighbor variable"
   print ("Sweep min neighbor start.")          
   for item in min_neighbors:
      faceMatches = []
      t_start = cv2.getTickCount()
      for index in range(len(faceGroup[group])):
          rects, img = detectFace(faceGroup[group][index], cascade, minNeighbors=item)
          faceMatches.append(len(rects))
      t_stop = cv2.getTickCount()
      t_total = (t_stop - t_start) / cv2.getTickFrequency()
      SweepTimeNeighbors.append(t_total)
                    
      # Compare accuracy
      accuracy = ComputeFaceAccuracy(faceMatches, faceGroup[group], faceTable)
      SweepAccuracyNeighbors.append(accuracy)
   print ("Sweep min neighbor complete.")                          
            
   # Sweep the "min size" variable
   print ("Sweep min size start.")               
   for item in min_size:
      faceMatches = []
      t_start = cv2.getTickCount()
      for index in range(len(faceGroup[group])):
          rects, img = detectFace(faceGroup[group][index], cascade, minSize=item)
          faceMatches.append(len(rects))
      t_stop = cv2.getTickCount()
      t_total = (t_stop - t_start) / cv2.getTickFrequency()
      SweepTimeMinSize.append(t_total)
                    
      # Compare accuracy
      accuracy = ComputeFaceAccuracy(faceMatches, faceGroup[group], faceTable)
      SweepAccuracyMinSize.append(accuracy)
   print ("Sweep min size complete." )  
            
    
   # Sweep the "max size" variable
   print ("Sweep max size start.")       
   for item in max_size:
      faceMatches = []
      t_start = cv2.getTickCount()
      for index in range(len(faceGroup[group])):
          rects, img = detectFace(faceGroup[group][index], cascade, maxSize=item)
          faceMatches.append(len(rects))
      t_stop = cv2.getTickCount()
      t_total = (t_stop - t_start) / cv2.getTickFrequency()
      SweepTimeMaxSize.append(t_total)
      accuracy = ComputeFaceAccuracy(faceMatches, faceGroup[group], faceTable)
      SweepAccuracyMaxSize.append(accuracy)
   print ("Sweep max size complete.")   
            
# Print the accuracy
print ("Accuracy [Scale, minNeighbors, minSize, maxSize]")
print (SweepAccuracyScale)
print (SweepAccuracyNeighbors)
print (SweepAccuracyMinSize)
print (SweepAccuracyMaxSize)

# Print the execution time
print ("Execution time [Scale, minNeighbors, minSize, maxSize]")
print (SweepTimeScale)
print (SweepTimeNeighbors)
print (SweepTimeMinSize)
print (SweepTimeMaxSize)
