
#MNIST Handwritten digits recognition using Keras (3 Layers Neural Network) 


# STEP 1 IMPORT THE NECESSARY PACKAGES

import keras
from keras.datasets import mnist
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.optimizers import SGD
from keras.utils import np_utils
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import numpy as np


# STEP 2 GRAB MNIST DATASET

# Read/Download MNIST Dataset
print('Loading MNIST Dataset...')
dataset = fetch_mldata('MNIST Original')

# Number of pixels
img_rows, img_columns = 28, 28

num_pixels = img_rows*img_columns

# Reshape the MNIST data: flatten it.  
#mnist_data = dataset.data.reshape((dataset.data.shape[0], num_pixels))


# Scale MNIST DATA 
# pixel values are gray scale bw 0 and 255
# we normalize pixel values to range 0 and 1 by dividing each value by max of 255
# Convert all the data into float32 for uniformity

#mnist_data = mnist_data.astype("float32")
#mnist_data = mnist_data/255.0


# Divide data into testing and training sets.
train_img, test_img, train_labels, test_labels = train_test_split(dataset.data,dataset.target.astype("int"), test_size=0.25)

# Reshape the MNIST data: flatten it.  
train_img = train_img.reshape(train_img.shape[0],num_pixels )
test_img = test_img.reshape(test_img.shape[0], num_pixels)

# Scale MNIST DATA 
# pixel values are gray scale bw 0 and 255
# we normalize pixel values to range 0 and 1 by dividing each value by max of 255
# Convert all the data into float32 for uniformity

train_img = train_img.astype("float32")
test_img = test_img.astype("float32")

train_img /= 255.0
test_img /= 255.0

print("\nINFORMATION:\n")
print('\nTrain image shape:{}'.format(train_img.shape))
print('\nTest image shape:{}'.format(test_img.shape))
print('\nTrain Samples :{} '.format(train_img.shape[0]))
print('\nTest Samples :{}'.format(test_img.shape[0] ))

# Total number of epochs
numb_epochs = 25

# Transform training and testing data to 10 classes in range [0,classes] ; num. of classes = 0 to 9 = 10 classes
totalClasses = 10			# 0 to 9 labels
train_labels = np_utils.to_categorical(train_labels, totalClasses)
test_labels = np_utils.to_categorical(test_labels, totalClasses)

# creating the model 
    
model = Sequential()
model.add(Dense(256,activation = 'relu',input_dim= num_pixels))
model.add(Dense(128,activation = 'relu'))
#model.add(Dense(128,activation = 'relu'))
model.add(Dense(totalClasses,activation = 'softmax'))


# Summarize the model

model.summary()

# compile the model
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer ="sgd", metrics=["accuracy"]) 

# fit the model
   
history = model.fit(train_img,train_labels,validation_data=(test_img,test_labels),batch_size = 128, epochs = numb_epochs,verbose = 0)    

#history = result

print(history.history.keys())

# Evaluate 

score = model.evaluate(test_img,test_labels, batch_size = 128, verbose = 0)
print("\nTEST SCORES\n")
print('\nTest loss:', score[0])
print('\nTest accuracy:', score[1])

predictions = model.predict(test_img,batch_size=128)

# Classification report
print("\n Classification Report: \n")
print(classification_report(test_labels.argmax(axis=1),predictions.argmax(axis=1)))


print("\n Plotting few images from test data with their predictions\n")
# Plot few test images and print the predictions 
# Grabbing few test images
testimg = test_img[5:9]

# Reshaping test image to [28x28] pixel inorder to plot them 
testimg = testimg.reshape(testimg.shape[0],28,28)

for i, testimg in enumerate(testimg, start = 1):
    testImage = testimg
    # Reshaping the image to 784 pixel intensties to find the prediction
    testimg = testimg.reshape(1,784)
    # Computing the prediction using keras function predict_classes
    prediction = model.predict_classes(testimg)
    
    # Print the predicted results 
    print("The handwritten digit can be {}".format(prediction[0]))
    
    # Plotting the image with subplots
    plt.subplot(220+i)
    plt.imshow(testImage, cmap='gray')
    
plt.show()

print("\n Plotting the variation in accuracy of our model with each epochs\n")
# history plot for accuracy
plt.plot(history.history["acc"], label = 'Train Accuracy')
plt.plot(history.history["val_acc"], label = 'Test Accuracy' )
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()



print("Plotting the variation in loss of our model with each epochs\n")
# history plot for loss
plt.plot(history.history["loss"],label = 'Train Loss')
plt.plot(history.history["val_loss"],label = 'Test loss')
plt.title("Model loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
