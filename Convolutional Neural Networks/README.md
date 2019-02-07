# Comparison of Convolutional Neural Networks 

## AlexNet, Inception_V3, resnet50, MobileNet, Xception, vgg19, vgg16

## flower17 Dataset

Lets compare the performance of various Neural Network. Flower17 data set is used for classification using various Convolutional Neural Network. 


## Data-set
We have a 17 category flower dataset with 80 images for each class. The flowers chosen are some common flowers in the UK. The images have large scale, pose and light variations and there are also classes with large varations of images within the class and close similarity to other classes. We randomly split the dataset into 3 different training, validation and test sets. A subset of the images have been groundtruth labelled for segmentation.


## Aim:
Design a 17-class image classifier based on a very small flower dataset.

1. Download the “flowers17" dataset from http://www.robots.ox.ac.uk/~vgg/data/flowers/17/

2. Use AlexNet as our first CNN for testing.

3. Design AlexNet and directly train on the flowers17 dataset.

4. Use data augmentation method to improve your accuracy. Please use “ImageDataGenerator” which is a Keras function.

5. Use transfer learning to further improve the accuracy using transfer learning.

6. Train the same data using Inception_V3, resnet50, MobileNet, Xception, vgg19, vgg16.

7. Plot the curves for Loss vs Epochs and Accuracy vs Epochs.

