# Deep Neural Network


## AIM : 

Build a Deep Neural Network on Python and C from scratch 

Execute the same on Raspberry PI

## [Task] Python

Multilayer implementation. Provided a single hidden layer code that uses random numbers for biases and weights.  Must be generalized to multi-stages.

Must take in a network description example:  example_Input2_Hidden_2x3_Output_4.txt

Example description file for 2 input nodes, 3 hidden nodes per 2 layers, and 4 output nodes

Input Nodes : 2

Hidden Layers : 2

Hidden Nodes : 3

Output Nodes : 4

HL_Weight : 0 : 4 5 6 : 7 8 9

HL_Bias : 0 : 1 2 3

HL_Weight : 1 : -1 0 1 : 2 1 2 : 3 2 1

HL_Bias : 1 : 1 2 3

Output_Weight : 0 : 6 5 3 1 : 0 2 1 2 : 8 9 -3 -4

Output_Bias : 0 : -1 0 1


Example HL_Weight : Level : weights between node 0 previous level and each input node : weights between node 1 and each input node

Example HL_Bias : Level : bias values for each node in layer

Level positions for Bias and Weight of Output are set to 0

###############################################################################

DNN is implemented using runExp.py executing all the (NEW) given 27 network and input files. The runExp.py runs through all the networks one by one and printing the results into a file name Results.txt. Data from Results.txt is used to plot the graphs to come to conclusion regarding the execution time and size of the neural network.

## Program Description: 

runExp.py can be used to run the DNN.c 
C code to execute all the 27 network files.
To execute the runExp.py,use runExp.py along with folder named DEMO, Inputs, Networks, DNN.c file on to server. 

And Use the following commands.

gcc DNN.c -lm ( to compile the DNN.c C code ) 

python runExp.py ( to run the runExp.py )


## [Task] C_Implementation

Multilayer implementation. Provided a single hidden layer code that uses random numbers for biases and weights.  Must be generalized to multi-stages.


## To execute this C code:

Use DNN.c  and the folder named Inputs, Networks, DEMO on to the server. And follow the commands below,

## Commands: 

gcc DNN.c -lm

./a.out Inputs/Input_4.in Networks/Networks_I4_H16x16_O4.net Cresults.txt.

Similarly change the name of input file and network file to run any of it.








