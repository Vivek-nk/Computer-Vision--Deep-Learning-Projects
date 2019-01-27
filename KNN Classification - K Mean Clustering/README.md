Implementing KNN (Classification) and Kmean (Clustering) solutions in Python. KNN will be performed using real data from the David Lowe SIFT algorithm (“Object recognition from local scale-invariant features," International Conference on Computer Vision, 1999), while Kmeans will be implemented with random data.  

Reading background: 

http://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm
http://en.wikipedia.org/wiki/Euclidean_distance


Background: Knn

Scale-invariant feature transform (or SIFT) is an algorithm in computer vision to detect and describe local features in images. The algorithm was published by David Lowe in 1999.
Applications include object recognition, robotic mapping and navigation, image stitching, 3D modeling, gesture recognition, video tracking, and match moving.
For any object in an image, interesting points on the object can be extracted to provide a "feature description" of the object. This description, extracted from a training image, can then be used to identify the object when attempting to locate the object in a test image containing many other objects. To perform reliable recognition, it is important that the features extracted from the training image be detectable even under changes in image scale, noise and illumination. Such points usually lie on high-contrast regions of the image, such as object edges.
Another important characteristic of these features is that the relative positions between them in the original scene shouldn't change from one image to another. For example, if only the four corners of a door were used as features, they would work regardless of the door's position; but if points in the frame were also used, the recognition would fail if the door is opened or closed. Similarly, features located in articulated or flexible objects would typically not work if any change in their internal geometry happens between two images in the set being processed. However, in practice SIFT detects and uses a much larger number of features from the images, which reduces the contribution of the errors caused by these local variations in the average error of all feature matching errors.

More information on David Lowe’s original algorithm: 

http://www.cs.ubc.ca/~lowe/keypoints/


Background: Kmeans

Given a set of observations (x1, x2, …, xn), where each observation is a d-dimensional real vector, k-means clustering aims to partition the n observations into k sets (k ≤ n) or classes. In each set S = {S1, S2, …, Sk}, the aim is to minimize the within-cluster sum of squares (WCSS):


Background: Data Format

You have access to four Scale Invariant Feature Transform (SIFT) descriptors.  Each SIFT descriptor consists of 5 descriptor identifiers (ID, X, Y, magnitude, orientation) and 128 keypoints.  The example data format of each file appears as below:

579,128
0,87.669998,55.509998,22.070000,0.024000,8,0,0,0,0,0,0,9,56,0,0,0,0,21,24,97,1,0,0,0,2,97,40,20,1,0,0,0,5,9,37,52,43,10,0,0,0,0,0,7,161,49,0,1,11,41,28,94,57,5,0,9,91,161,62,71,17,4,0,2,80,38,73,107,52,2,0,0,0,0,0,11,161,28,7,34,18,2,0,55,67,65,33,158,111,13,1,9,25,112,29,13,84,23,1,10,12,0,0,0,0,0,0,0,87,2,1,10,5,0,0,4,4,15,11,46,19,0,0,0,5,47,10,3,8,1,0,0
1,143.059998,31.420000,13.570000,0.054000,23,10,0,0,0,0,0,0,159,97,0,0,0,0,0,10,159,36,2,7,100,22,9,16,30,8,1,7,159,27,2,8,25,1,0,0,0,0,0,6,159,27,0,0,0,0,0,85,159,43,39,90,70,0,0,19,8,22,17,121,122,1,0,0,15,4,0,0,0,0,0,0,159,67,0,0,1,0,0,7,73,19,3,36,91,0,0,1,0,1,6,57,72,0,0,0,0,0,0,0,0,0,0,0,7,1,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0

The first line in each file consists of the number of descriptors and the length of the keypoint.  The descriptor 0 above has the following expressed meaning:

0,87.669998,55.509998,22.070000,0.024000,8,0,0,0,0,0,…

ID: 0
X position in image: 87.669998 
Y position in image: 55.509998
Magnitude: 22.070000
Orientation: 0.024000
128 values of keypoints: 8,0,0,0,0,0, …

Note that the format of each line is CSV (Comma separated version) form.

You have four keypoint files: Basmati.key, Book.key, Box.key, Scene.key

Each data file is read into a numpy array. 

KNN matching between SIFT keypoint files: 

Items: 

•	Load each of the 4 keypoint files into 4 separate numpy arrays.  Arrays should be 2D in that the number of rows is the number of keypoints and the number of columns is 128 values.

•	Construct a KNN Python function that compares the 2 keypoint arrays. Example uses will be:

KNNmatch(BasmatiKeys, SceneKeys,k=2, confidence=50)
KNNmatch(BoxKeys, SceneKeys, k=2, confidence=50)
KNNmatch(BookKeys, SceneKeys, k=2, confidence=50)

The goal of the KNNmatch function is to process every keypoint in the target array to the scene array.
For only k=2, the top two matches in the scene keypoints will be calculated.  The threshold score will determine if there is an accuracy match between the 2 nearest neighbor keypoints.

There is a match if the distance between the target keypoint and the two nearest neighbor scene keypoints.   

Min1 = distance(target keypoint, 1st nearest scene keypoint)
Min2 = distance(target keypoint, 2nd nearest scene keypoint)

if( 100 * min1 <  confidence * min2)

   validMatch++

We are using the term confidence, but it really serves as a threshold. As a note, a confidence score of 50 essentially determines that the nearest neighbor match in the scene in 50% better than then next nearest match. A confidence setting of 100 states that min1 just has to be below min2. The lower the confidence, for example a setting of 10, would say that a nearest match in the scene only has to be more strict and 90% better than the next nearest match. There’s essentially as 100-confidence to get the percent smaller that min1 must be to min2. 

Using KNNmatch function,the confidence scores is compared in descending order from: [10, 20, 30, 40, 50]. 
In descending order, it is assumed that the level of confidence of 50 provides the most accurate set of keypoints. 
Plotting the number of keypoints that are deemed matches for Box, Book, Basmati for the confidence.  



