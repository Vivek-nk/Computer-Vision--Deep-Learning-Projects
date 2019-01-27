import os
from  pylab import *

# Function to Perform KNNmatch algorithm on data
def KNNmatch(data1, data2, k, confidence_list, return_sum):
    score = zeros((len(data1)))
    match = zeros((len(data2), len(confidence_list)))
    for i in range(len(data2)):
        for j in range(len(data1)):
            score[j] = sum(square(data1[j,:] - data2[i,:]))
        imin0 = argmin(score)
        min0 = score[imin0]
        score[imin0] = nan
        min1 = nanmin(score)
        for n in arange(len(confidence_list)):
            if min0 < float(confidence_list[n]) / 100 * min1:
                match[i,n] = 1
    if return_sum:
        return sum(match, axis = 0)
    return match

    
# Load in the keypoint files and run the KNNmatch
# Funcion on each target
# Compare each target to the scene over
# for multiple confidence values
k = 2
confidence_list = arange(1,11) * 10
mycols = range(5,128+5)

# Enter Correct path to the folder containing the *.key Files
path = "DataKey"
fileList = ["Book.key", "Box.key", "Basmati.key", "Scene.key"]

storage = dict()
for name in fileList:
    file = os.path.join(path, name)
    storage[name] = loadtxt(fname=file, delimiter=",", skiprows=1,usecols=mycols)

book_match = KNNmatch(storage[fileList[0]], storage[fileList[3]], k, confidence_list,1)
box_match = KNNmatch(storage[fileList[1]], storage[fileList[3]], k, confidence_list, 1)
basmati_match = KNNmatch(storage[fileList[2]], storage[fileList[3]], k, confidence_list, 1)


# Plot the results of the KNNmatch function for various confidence values
thresh = 100 - confidence_list
legend_name = ['Book', 'Box', 'Basmati']
figure(figsize = (17,8), dpi = 500)
plot(thresh, book_match, label = legend_name[0], linewidth = 2)
plot(thresh, box_match, label = legend_name[1], linewidth = 2)
plot(thresh, basmati_match, label = legend_name[2], linewidth = 2)
legend(loc = 'upper right')
xlabel('Keypoint Threshold Level')
ylabel('Number of Keyspoints')
title('Keypoint Threshold Vs. Number of Matches')

