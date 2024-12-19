import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data = pd.read_csv('framingham.csv', delimiter=',')
#cleaned data D
dataOriginal = data.dropna()
#data Normalized
D = dataOriginal/ dataOriginal.max()
#weights initial
w = np.random.uniform(-4,4,15)
c= 3*np.random.rand()*0

#trial and error rate
learningRate = 4
temp=100
cost=10
#runs until cost reductions are of order 1e-6
y=D.iloc[:,15]
yPrediction=[]
while temp-cost>1e-6:

    #prediction
    sum = c
    for j in range(15):
        sum+= w[j]*D.iloc[:,j]
    expon = np.exp(-(sum))
    yPrediction = 1/(1+expon)

    #negative derivaitives
    dWeights = np.array([((D.iloc[:,idx]*expon/((1+expon)**2))*(y/yPrediction-(1-y)/(1-yPrediction))).mean() \
for idx,v in enumerate(w)])
    dc = ((expon/((1+expon)**2))*(y/yPrediction-(1-y)/(1-yPrediction))).mean()

    #change weights
    w += learningRate*dWeights
    c+= dc*learningRate
    temp=cost
    cost = -(y*np.log(yPrediction)+(1-y)*np.log(1-yPrediction)).mean()

yPrediction[yPrediction>0.5]=1
yPrediction[yPrediction!=1]=0
totalNum = len(yPrediction)
#MAKE BINARY
y=np.round(y).astype(int)
yPrediction=np.round(yPrediction).astype(int)

numWrong = (y^yPrediction).sum()
numCorrect = totalNum - numWrong
numPositveCorrect = (y&yPrediction).sum()
numNegativeCorrect = numCorrect - numPositveCorrect
x = (yPrediction-y)
x[x!=1]=0
numFalsePositive = x.sum()
numFalseNegative = yPrediction.sum()-numFalsePositive

print("percentage of data points classified correctly: "+str(np.round(numCorrect/totalNum*100,2)))
print("percentage false negatives (numbers of false negatives /number of positives):"+str(np.round(numFalseNegative/(yPrediction.sum())*100,2)))
print("percentage false positives(number of false positives/number of negatives)"+str(np.round(numFalsePositive/((totalNum-yPrediction.sum()))*100,2)))
