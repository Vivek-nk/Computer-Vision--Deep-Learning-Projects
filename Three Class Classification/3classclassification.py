
# IMPORT ALL NECCESSARY LIBRARIES
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import glob
import operator
import math
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import  accuracy_score


# Full path of files
directoryPath = '/home/vivek/Downloads/Homework_1_export/animals'

# Initializing lists to store file name, images, labels and data 
fileNameList = []
imageList = []
labels = []
data = []

# Walk through directory path to access all the files & folders & subfolders
for dirpath, dirnames, filenames in os.walk(directoryPath):
    # Save the path to all files in Imagefile
    
    for filename in filenames:
        imageFile = (os.path.join(dirpath, filename))
        fileNameList.append(filename)
        #print(imageFile)

        
        # Read OpenCV image & append the matrix into a image list
        img = cv2.imread(imageFile)
        imageList.append(img)
        
        # Split the file name to get the label & append to a list
        label = filename.split("_")[0]
        labels.append(label)
        
        # Resize each image to (32,32) size and convert the matrix into a data vector
        width = 32
        height = 32
        dim = (width, height)
        resized_Image = cv2.resize(img, dim, interpolation = cv2.INTER_AREA).flatten()
        
        # append the first letter of the label to the end of each array in data vector 
        # d for dogs, c for cats & p for panda
        resized_Image = np.append(resized_Image, label[0])
        
        # append the data vector into a Data List
        data.append(resized_Image)
        

# Shuffling the data before splitting 
np.random.shuffle(data)


# Splitting the data set into Test set , Train set & Validate set 

trainData, validateData, testData = np.split(data, [int(.7* len(data)), int(.8* len(data))])


# Creating y_test, y_train from our trainData & testData

y_train = []

for index in range(len(trainData)):
    
    ytrain = trainData[index][-1]
    y_train.append(ytrain)
    
y_test = []

for index in range(len(testData)):
    
    ytest = testData[index][-1]
    y_test.append(ytest)


# Print Test Data: 

print("\nTest Data:\n", testData)

# Print Train Data: 

print("\nTrain Data:\n", trainData)

# Print Validate Data: 

print("\nValidate Data:\n", validateData)

#print y_test & y_train 

print("\nActual Value Y_test:\n", y_test)

print("\nActual Value Y_train:\n", y_train)


# Printing the information about each data set 

print("Printing Information..")

# Size of data
print("\nLength of entire Data Set is :",len(data))

# Size of Train, Test, Validate Set
print("Length of Train Data set:",len(trainData),"\nLength of Validate Data set:", len(validateData),"\nLength of Test Data Set:",len(testData))

# Size of actual value of test & train 
print("\nLength of Y_test:", len(y_test))
print("\nLength of Y_train:", len(y_train))


# Function to calculate L1 (Manhattan) Distance
def manhattanDistance(index1, index2, limit):
    distance = 0
    for ind in range(limit):
        distance += (float(index1[ind]) - float(index2[ind]))
    return distance


# Function to calculate L2 (Eculidean) Distance
def euclideanDistance(index1, index2, limit):
    distance = 0
    for ind in range(limit):
        distance += pow((float(index1[ind]) - float(index2[ind])), 2)
    return math.sqrt(distance)


# Function to get the neighbors 
# Funtion to return the K most similar instance from training set for a given test set 

def KNN_Neighbors(trainSet, testInstance, k):
    distances = []
    limit = len(testInstance)-1
    for index in range(len(trainSet)):
        #dist = manhattanDistance(testInstance, trainSet[index], limit)
        dist = euclideanDistance(testInstance, trainSet[index], limit)
        distances.append((trainSet[index], dist))
     
    #SORT THE DISTANCES:- 
    # index 1 is the calculated distance between training_instance and test_instance
    #sorted_distances = sorted(distances, key=itemgetter(1))
    distances.sort(key=operator.itemgetter(1))
    
    neighbors = []
    # Extracting K Nearest Neighbors and append into list 
    for i in range(k):
        neighbors.append(distances[i][0])
    
    # For calculating most frequent class amoung neighbors
    Class_Votes = {}

    for index in range(len(neighbors)):
        
        # Extract the last element which is the label
        result = neighbors[index][-1]
    
        # Counting the number of label
        if result in Class_Votes:
        
            Class_Votes[result] += 1
        
        else:
        
            Class_Votes[result] = 1
    
    # Compute the maximum vote by sorting 
    Sorted_Votes = sorted(Class_Votes.items(), key = operator.itemgetter(1), reverse = True)
        
    return Sorted_Votes[0][0], neighbors


# Function to calculate accuracy of the model 
def calculate_accuracy(testSet, predictions):
    
    correctPrediction = 0
    
    for i in range(len(testSet)):
        
        if testSet[i][-1] == predictions[i]:
            
            correctPrediction += 1
                 
    return (correctPrediction/float(len(testSet))) * 100


# Value of K = 3 
k=3
predictions_k3 = []
# Predicting the most probable result for 3 nearest neighbors 
for index in range(len(testData)):
    predicted, neighbors = KNN_Neighbors(trainData, testData[index], k)
    predictions_k3.append(predicted)
    print("Predicted Value = "+ predicted + " and Actual Value = " + testData[index][-1])
    print("Predictied Neighbors Data Vector : ", neighbors)
    print("\n")


# Calculating accuracy using my own function for k =3
accuracy_k3 = calculate_accuracy(testData, predictions_k3)
print("\nThe overall accuracy of our model is:",accuracy_k3)

# Printing Confusion Matrix and Classification Report for k =3
print("\nConfusion Matrix:\n",confusion_matrix(y_test, predictions_k3))  
print("\nClassification report:\n",classification_report(y_test, predictions_k3)) 

# Calculating accuracy using function accuracy_score from sklearns for k =3
accuracy_k3 = accuracy_score(y_test, predictions_k3) * 100
print('\nThe overall accuracy of our model is using accuracy_score:',accuracy_k3)


confusionMatk3 = confusion_matrix(y_test, predictions_k3)
ax = sns.heatmap(confusionMatk3,annot = True, fmt ='d' )
                                  


# Value of K = 5
k=5
predictions_k5 = []
# Predicting the most probable result for 5 nearest neighbors 
for index in range(len(testData)):
    predicted, neighbors = KNN_Neighbors(trainData, testData[index], k)
    predictions_k5.append(predicted)
    print("Predicted Value = "+ predicted + " and Actual Value = " + testData[index][-1])
    print("Predictied Neighbors Data Vector : ", neighbors)
    print("\n")


# Calculating accuracy using my own function for k =5
accuracy_k5 = calculate_accuracy(testData, predictions_k5)
print("\nThe overall accuracy of our model when k =5 is:",accuracy_k5)

# Printing Confusion Matrix and Classification Report for k =5
print("\nConfusion Matrix:\n",confusion_matrix(y_test, predictions_k5))  
print("\nClassification report:\n",classification_report(y_test, predictions_k5)) 

# Calculating accuracy using function accuracy_score from sklearns for k =5
accuracy_k5 = accuracy_score(y_test, predictions_k5) * 100
print('\nThe overall accuracy of our model using accuracy_score when k =5 :',accuracy_k5)

# Plotting the confusion matrix using seaborn heatmap
confusionMatk5 = confusion_matrix(y_test, predictions_k5)
ax = sns.heatmap(confusionMatk5, annot = True, fmt ='d' )
                                  

# Value of K = 7
k=7
predictions_k7 = []
# Predicting the most probable result for 7 nearest neighbors 
for index in range(len(testData)):
    predicted, neighbors = KNN_Neighbors(trainData, testData[index], k)
    predictions_k7.append(predicted)
    print("Predicted Value = "+ predicted + " and Actual Value = " + testData[index][-1])
    print("Predictied Neighbors Data Vector : ", neighbors)
    print("\n")

# Calculating accuracy using my own function for k = 7
accuracy_k7 = calculate_accuracy(testData, predictions_k7)
print("\nThe overall accuracy of our model when k = 7 is:",accuracy_k7)

# Printing Confusion Matrix and Classification Report for k = 7
print("\nConfusion Matrix:\n",confusion_matrix(y_test, predictions_k7))  
print("\nClassification report:\n",classification_report(y_test, predictions_k7)) 

# Calculating accuracy using function accuracy_score from sklearns for k = 7
accuracy_k7 = accuracy_score(y_test, predictions_k7) * 100
print('\nThe overall accuracy of our model using accuracy_score when k = 7 is:',accuracy_k7)

# Plotting the confusion matrix using seaborn heatmap
confusionMatk7 = confusion_matrix(y_test, predictions_k7)
ax = sns.heatmap(confusionMatk5, annot = True, fmt ='d' )


# Value of K = 9
k=9
predictions_k9 = []
# Predicting the most probable result for 9 nearest neighbors 
for index in range(len(testData)):
    predicted, neighbors = KNN_Neighbors(trainData, testData[index], k)
    predictions_k9.append(predicted)
    print("Predicted Value = "+ predicted + " and Actual Value = " + testData[index][-1])
    print("Predictied Neighbors Data Vector : ", neighbors)
    print("\n")



# Calculating accuracy using my own function for k = 9
accuracy_k9 = calculate_accuracy(testData, predictions_k9)
print("\nThe overall accuracy of our model when k = 9 is:",accuracy_k9)

# Printing Confusion Matrix and Classification Report for k = 9
print("\nConfusion Matrix:\n",confusion_matrix(y_test, predictions_k9))  
print("\nClassification report:\n",classification_report(y_test, predictions_k9)) 

# Calculating accuracy using function accuracy_score from sklearns for k = 9
accuracy_k9 = accuracy_score(y_test, predictions_k9) * 100
print('\nThe overall accuracy of our model when k = 9 using accuracy_score is:',accuracy_k9)

# Plotting the confusion matrix using seaborn heatmap
confusionMatk9 = confusion_matrix(y_test, predictions_k9)
ax = sns.heatmap(confusionMatk5, annot = True, fmt ='d' )


# Value of K = 11
k=11
predictions_k11 = []
# Predicting the most probable result for 11 nearest neighbors 
for index in range(len(testData)):
    predicted, neighbors = KNN_Neighbors(trainData, testData[index], k)
    predictions_k11.append(predicted)
    print("Predicted Value = "+ predicted + " and Actual Value = " + testData[index][-1])
    print("Predictied Neighbors Data Vector : ", neighbors)
    print("\n")


# Calculating accuracy using my own function for k = 11
accuracy_k11 = calculate_accuracy(testData, predictions_k11)
print("\nThe overall accuracy of our model when k = 11 is:",accuracy_k11)

# Printing Confusion Matrix and Classification Report for k = 11
print("\nConfusion Matrix:\n",confusion_matrix(y_test, predictions_k11))  


# Plotting the confusion matrix using seaborn heatmap
confusionMatk11 = confusion_matrix(y_test, predictions_k11)
ax = sns.heatmap(confusionMatk5, annot = True, fmt ='d' )


print("\nClassification report:\n",classification_report(y_test, predictions_k11))

# Calculating accuracy using function accuracy_score from sklearns for k = 11
accuracy_k11 = accuracy_score(y_test, predictions_k11) * 100
print('\nThe overall accuracy of our model when k = 11 using accuracy_score:',accuracy_k11)



# Value of K = 13
k=13
predictions_k13 = []
# Predicting the most probable result for 13 nearest neighbors 
for index in range(len(testData)):
    predicted, neighbors = KNN_Neighbors(trainData, testData[index], k)
    predictions_k13.append(predicted)
    print("Predicted Value = "+ predicted + " and Actual Value = " + testData[index][-1])
    print("Predictied Neighbors Data Vector : ", neighbors)
    print("\n")


# Calculating accuracy using my own function for k = 13
accuracy_k13 = calculate_accuracy(testData, predictions_k13)
print("\nThe overall accuracy of our model when k = 13 is:",accuracy_k13)

# Printing Confusion Matrix and Classification Report for k = 13
print("\nConfusion Matrix:\n",confusion_matrix(y_test, predictions_k13))  
print("\nClassification report:\n",classification_report(y_test, predictions_k13)) 

# Calculating accuracy using function accuracy_score from sklearns for k = 13
accuracy_k13 = accuracy_score(y_test, predictions_k13) * 100
print('\nThe overall accuracy of our model when k = 13 using accuracy_score:',accuracy_k13)



# Plotting the confusion matrix using seaborn heatmap
confusionMatk13 = confusion_matrix(y_test, predictions_k13)
ax = sns.heatmap(confusionMatk5, annot = True, fmt ='d' )


# Value of K = 15
k=15
predictions_k15 = []
# Predicting the most probable result for 15 nearest neighbors 
for index in range(len(testData)):
    predicted, neighbors = KNN_Neighbors(trainData, testData[index], k)
    predictions_k15.append(predicted)
    print("Predicted Value = "+ predicted + " and Actual Value = " + testData[index][-1])
    print("Predictied Neighbors Data Vector : ", neighbors)
    print("\n")


# Calculating accuracy using my own function for k = 15
accuracy_k15 = calculate_accuracy(testData, predictions_k15)
print("\nThe overall accuracy of our model when k = 15 is:",accuracy_k15)

# Printing Confusion Matrix and Classification Report for k = 15
print("\nConfusion Matrix:\n",confusion_matrix(y_test, predictions_k15))  
print("\nClassification report:\n",classification_report(y_test, predictions_k15)) 

# Calculating accuracy using function accuracy_score from sklearns for k = 15
accuracy_k15 = accuracy_score(y_test, predictions_k15) * 100
print('\nThe overall accuracy of our model when k = 15 using accuracy_score is:',accuracy_k15)


# Plotting the confusion matrix using seaborn heatmap
confusionMatk15 = confusion_matrix(y_test, predictions_k15)
ax = sns.heatmap(confusionMatk5, annot = True, fmt ='d' )

# Value of K = 21
k=21
predictions_k21 = []
# Predicting the most probable result for 21 nearest neighbors 
for index in range(len(testData)):
    predicted, neighbors = KNN_Neighbors(trainData, testData[index], k)
    predictions_k21.append(predicted)
    print("Predicted Value = "+ predicted + " and Actual Value = " + testData[index][-1])
    print("Predictied Neighbors Data Vector : ", neighbors)
    print("\n")


# Calculating accuracy using my own function for k = 21
accuracy_k21 = calculate_accuracy(testData, predictions_k21)
print("\nThe overall accuracy of our model when k = 21 is:",accuracy_k21)

# Printing Confusion Matrix and Classification Report for k = 21
print("\nConfusion Matrix:\n",confusion_matrix(y_test, predictions_k21))  
print("\nClassification report:\n",classification_report(y_test, predictions_k21)) 

# Calculating accuracy using function accuracy_score from sklearns for k = 21
accuracy_k21 = accuracy_score(y_test, predictions_k21) * 100
print('\nThe overall accuracy of our model when k = 21 using accuracy_score is:',accuracy_k21)


# Plotting the confusion matrix using seaborn heatmap
confusionMatk21 = confusion_matrix(y_test, predictions_k21)
ax = sns.heatmap(confusionMatk5, annot = True, fmt ='d' )


# Value of K = 29
k=29
predictions_k29 = []
# Predicting the most probable result for 29 nearest neighbors 
for index in range(len(testData)):
    predicted, neighbors = KNN_Neighbors(trainData, testData[index], k)
    predictions_k29.append(predicted)
    print("Predicted Value = "+ predicted + " and Actual Value = " + testData[index][-1])
    print("Predictied Neighbors Data Vector : ", neighbors)
    print("\n")

# Calculating accuracy using my own function for k = 29
accuracy_k29 = calculate_accuracy(testData, predictions_k29)
print("\nThe overall accuracy of our model when k = 29 is:",accuracy_k29)

# Printing Confusion Matrix and Classification Report for k = 29
print("\nConfusion Matrix:\n",confusion_matrix(y_test, predictions_k29))  
print("\nClassification report:\n",classification_report(y_test, predictions_k29)) 

# Calculating accuracy using function accuracy_score from sklearns for k = 29
accuracy_k29 = accuracy_score(y_test, predictions_k29) * 100
print('\nThe overall accuracy of our model when k = 29 using accuracy_score is:',accuracy_k29)


# Plotting the confusion matrix using seaborn heatmap
confusionMatk29 = confusion_matrix(y_test, predictions_k29)
ax = sns.heatmap(confusionMatk5, annot = True, fmt ='d' )


# Value of K = 31
k=31
predictions_k31 = []
# Predicting the most probable result for 31 nearest neighbors 
for index in range(len(testData)):
    predicted, neighbors = KNN_Neighbors(trainData, testData[index], k)
    predictions_k31.append(predicted)
    print("Predicted Value = "+ predicted + " and Actual Value = " + testData[index][-1])
    print("Predictied Neighbors Data Vector : ", neighbors)
    print("\n")


# Calculating accuracy using my own function for k = 31
accuracy_k31 = calculate_accuracy(testData, predictions_k31)
print("\nThe overall accuracy of our model when k = 31 is:",accuracy_k31)

# Printing Confusion Matrix and Classification Report for k = 31
print("\nConfusion Matrix:\n",confusion_matrix(y_test, predictions_k31))  
print("\nClassification report:\n",classification_report(y_test, predictions_k31)) 

# Calculating accuracy using function accuracy_score from sklearns for k = 31
accuracy_k31 = accuracy_score(y_test, predictions_k31) * 100
print('\nThe overall accuracy of our model when k = 31 using accuracy_score is:',accuracy_k31)


# Plotting the confusion matrix using seaborn heatmap
confusionMatk31 = confusion_matrix(y_test, predictions_k31)
ax = sns.heatmap(confusionMatk5, annot = True, fmt ='d' )



# Graph: Accuracy Vs KValue

Accuracy = [accuracy_k3,accuracy_k5,accuracy_k7,accuracy_k9,accuracy_k11,accuracy_k13,accuracy_k15,accuracy_k21,accuracy_k29,accuracy_k31]
print(Accuracy)
KValue = [3,5,7,9,11,13,15,21,29,31]
plt.plot(KValue, Accuracy, color = 'orange')
plt.xlabel('K Value')
plt.grid(True)
plt.ylabel('Accuracy in %')
plt.title('Accuracy Vs K Value')
plt.show()


# Using Validate Data set and Value of K = 29
k=29
predictions_validateData = []

# Predicting the most probable result for 29 nearest neighbors 
for index in range(len(testData)):
    predicted, neighbors = KNN_Neighbors(validateData, testData[index], k)
    predictions_validateData.append(predicted)
    print("Predicted Value = "+ predicted + " and Actual Value = " + testData[index][-1])
    print("Predictied Neighbors Data Vector : ", neighbors)
    print("\n")


# Calculating accuracy using my own function for k = 29
accuracy_Validate = calculate_accuracy(testData, predictions_validateData)
print("\nThe overall accuracy of Validate data test when k = 29 is:",accuracy_Validate)

# Printing Confusion Matrix and Classification Report for k = 29
print("\nConfusion Matrix:\n",confusion_matrix(y_test, predictions_validateData))  
print("\nClassification report:\n",classification_report(y_test, predictions_validateData)) 

# Calculating accuracy using function accuracy_score from sklearns for k =29
accuracy_Validate = accuracy_score(y_test, predictions_validateData) * 100
print('\nThe overall accuracy of validate data set when k = 29 using accuracy_score is:',accuracy_Validate)


# Plotting the confusion matrix using seaborn heatmap
confusionMatrixVal = confusion_matrix(y_test, predictions_validateData)
ax = sns.heatmap(confusionMatrixVal, annot = True, fmt ='d' )

