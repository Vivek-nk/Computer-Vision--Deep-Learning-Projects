# Three Class Animal Classification using KNN 

## Abstract

KNN algorithm is the simplest classification algorithm. It is based on feature similarity. The output of KNN is a class membership, a predicted class. An object is classified by a majority vote of its neighbors, with the object being assigned to the class most common among its K nearest neighbors. 

## Quick Summary of KNN

1.	Determine parameter K = number of nearest neighbors
2.	Calculate the distance between the query instance and all the training samples. 
3.	Sort the distance and determine the nearest neighbors based on the K-th minimum distance. 
4.	Gather the category of the nearest neighbors. 
5.	Use simple majority of the category of nearest neighbors as the predictions value of the query instance. 

## INTRODUCTION 

In this project lets try a 3-Class animals classification using K-Nearest Neighbor classifier. We have an animal dataset consisting of 3,000 images with 1,000 images per dog, cat and panda class respectively. Each image is represented in the RGB color space. These images are preprocessed into 32X32 pixels. And then the data set of animals is split into three. One for training, one for validation and other for testing. These images are converted into data vectors with their labels. And then KNN classifier is trained and the best value of K is determined for the highest accuracy. After training the performance of the KNN classifier is evaluated. 


## Dataset 

Create a dataset in this format: 

animals/
       cats/
       dogs/
       pandas/
       
Sample dataset has been uploaded. It does not have all the images as its difficult to upload all the images. Use Kaggle to download any neccessary dataset. 
