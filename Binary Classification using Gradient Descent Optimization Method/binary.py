# import the necessary packages
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
        
    # compute the sigmoid activation value for a given input
    #sigmoid(x) = 1 / (1 + exp(-x))
            
    return 1/ (1 + np.exp(-x))


def predict(X, W):
    
    # take the dot product between our features (X) and weight matrix (W)    
    # Bias is already included to the end of Input Matrix X
    
    Z = X.dot(W) 
    
    #print("activation :", Z)
    
    # apply a step function to threshold (=0.5) the outputs to binary class labels
    
    preds = sigmoid(Z)
    
    predictions = list()
    
    for x in preds:
        
        if x > 0.5: 
            
            preds = 1
            
        else:
            
            preds = 0
    
        predictions.append(preds)
    
       
    return predictions


epochs = 50
alpha = 0.1

# generate a 2-class classification problem with 1,000 data points, where each data point is a 2D feature vector
# X: data
# y: label

(X,y) = make_moons(n_samples=1000, noise = 0.15)

y = y.reshape(y.shape[0],1)

print ("Generated Input: \n", X)
print(" \nThe integer label (0 or 1) for class membership of each sample:\n",y) 


# insert a column of 1's as the last entry in the feature matrix -- this little trick allows us to treat the bias
# as a trainable parameter within the weight matrix
X = np.c_[X, np.ones((X.shape[0]))]

# partition the data into training and testing splits using 50% of the data for training and the remaining 50% for testing
(trainX, testX, trainY, testY) = train_test_split(X, y,test_size = 0.5, random_state = 30)

#print(X.shape)

print("[INFO] training...")

print ("\nSize of trainX:", trainX.shape)
print ("\nSize of testX:", testX.shape)
print ("\nSize of trainY:", trainY.shape)
print ("\nSize of testY:", testY.shape)

# initialize our weight matrix 
Weights = np.random.randn(testX.shape[1], 1)

print("\nSize of Weight:", Weights.shape)



# Define our loss function:
def lossFunction(Yhat, Y):
    
    return (-Y * np.log(Yhat) - (1 - Y) * np.log(1- Yhat)).mean()



# Initialize list for losses
losses = []
# loop over the desired number of epochs

for epoch in np.arange(0, epochs):
    # take the dot product between our features `X` and the weight
    # matrix `W`, then pass this value through our sigmoid activation
    # function, thereby giving us our predictions on the dataset
      
    Z = trainX.dot(Weights) 
    
    Yhat = sigmoid(Z)
    
    #print("Prediction:\n", Yhat)
        
    #  now that we have our predictions, we need to determine the
    # `error`, which is the difference between our predictions and the true values
    # loss: loss value for each iteration
           
    loss = lossFunction(Yhat, trainX)
    losses.append(loss)

     
    #the gradient descent update is the dot product between our
    #features and the error of the predictions
    # slope of the cost function across all observations
    
    gradient = np.dot(trainX.T, (Yhat - trainY))
    
    # in the update stage, all we need to do is "nudge" the weight
    # matrix in the negative direction of the gradient (hence the
    # term "gradient descent" by taking a small step towards a set
    # of "more optimal" parameters
        
    N = trainY.shape[0]
          
    # Taking average cost derivative for each feature
    
    gradient /= N
    
    # multiply the gradient by the learning rate
    
    gradient *= alpha
    
    # subtract from our weights to minimize cost
    
    Weights -= gradient 

   
    #check to see if an update should be displayed
    if epoch == 0 or (epoch + 1) % 5 == 0:
        
        print("For EPOCH = {}, loss = {}".format(int(epoch+1),loss))


# evaluate our model
print("[INFO] evaluating...")
preds = predict(testX, Weights)
print("\nClassification Report\n:",classification_report(testY, preds))

accuracy = accuracy_score(testY, preds) * 100
print("\nAccuracy of our model is:", accuracy)

print("\nConfusion Matrix:\n",confusion_matrix(testY, preds)) 

# plot the (testing) classification data
plt.style.use("ggplot")
plt.figure()
plt.title("Data")
plt.scatter(testX[:, 0], testX[:, 1], marker="o", c=testY[:,0], s=30)

# construct a figure that plots the loss over time
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, epochs), losses)
plt.title("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()

