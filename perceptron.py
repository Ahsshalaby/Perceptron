
# This program is a perceptron Algorithm to Sort Testing and Training Data

import numpy as np

#training data
trainingData = np.loadtxt(r"G:\Other computers\My Laptop\New Data\Mason Semester 1 Coursework\cs 688 ML\perceptron project\set9.train")
x1Train = trainingData[:,0] # x1 is different coordinates
x2Train = trainingData[:,1] # x2 is different coordinates
yTrain = trainingData[:,2] # y is label

#testing data
testingData = np.loadtxt(r"G:\Other computers\My Laptop\New Data\Mason Semester 1 Coursework\cs 688 ML\perceptron project\set.test")
x1Test = testingData[:,0] # x1 is different coordinates
x2Test = testingData[:,1] # x2 is different coordinates
yTest = testingData[:,2] # y is label

#initialized perceptron weights. I did random weights at first like HW says, but you get very inconsistent results from that making it impossible to compare the different perceptrons. 
#since there is only 40 test points the intial weights used has a large impact
w1 = 5
w2 = 5
b = 1

# this is the percepton stepFunction aka the prediction when I input w1*x1 + w2*x2 + b
def stepFunction(x):
    if x>=0:
        return 1
    else:
        return 0

for j in range(2): # passing the training data twice
    for i in range(len(x1Train)): # for all training data points
        x1 = x1Train[i] # setting x1 , x2, and y properly
        x2 = x2Train [i]
        y = yTrain [i]

        if (stepFunction(w1*x1 + w2*x2 + b) > y): # aka step gave 1 and y is zero
            w1 = w1 - x1
            w2 = w2 - x2
            b = b - 1
        # update weights properly

        if (stepFunction(w1*x1 + w2*x2 + b) < y): # aka step gave 0 and y is one
            w1 = w1 + x1
            w2 = w2 + x2
            b = b + 1
        #update weights properly
    
    
    #print(x1 , x2 , y, stepFunction(w1*x1 + w2*x2 + b)) # seems to already work well for test function at least



# now we have our w1, w2, and b value from the above loop
# w1*x1 + w2*x2 + b is our decision boundry

errors = 0 # intialize number errors to 0

for i in range(len(yTest)):
    x1 = x1Test[i]
    x2 = x2Test [i]
    y = yTest [i]
    #print(x1 , x2 , y, stepFunction(w1*x1 + w2*x2 + b))
    if (y != stepFunction(w1*x1 + w2*x2 + b)): # comparing real label to step prediction. if wrong increase counter
        errors = errors + 1

print(w1,w2,b)
print(errors/(len(yTest)))