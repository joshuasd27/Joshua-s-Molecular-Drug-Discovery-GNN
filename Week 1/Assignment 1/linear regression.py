import numpy as np
import matplotlib.pyplot as plt

def gradient(func,args):
    gradientArray=np.empty(len(args))
    dx=0.00001
    output = func(*args)
    for index, parameter in enumerate(args):  
        args[index] += dx
        gradientArray[index] = (func(*args)-output)/dx   
        args[index] -= dx
    return gradientArray


class simpleLinearFitter:
    def __init__(self, input_data, output_data,iterations):
        self.x = input_data
        self.y = output_data
        self.iter = iterations
        #guess m,c on 2 rand data pts
        a,b = np.random.randint(0,x.size,2)
        m = (y[b]-y[a])/(x[b]-x[a])
        c = y[a]-m*x[a]
        self.parameters = {'m': m,'c': c}
    
    def prediction(*self.parameters):
        return m*self.x+c

    def costFunc(self,predictions):
        return ((self.y-predictions)**2).mean
    
    def totalFunc(self,m,c):
        return self.costfunc(self.prediction())
    
    def train(self):
        #for i in range(self.iter):
            cost = self.costfunc(self.prediction())
            print("Iteration:{}, m:{}, c:{}, cost:{}".format(0,self.parameters['m'],self.parameters['c'],cost))

            
            grad = gradient(self.totalFunc, self.parameters)
            print(grad)


            #change m,c
            stepSize = 0.03
            







#fig, (ax1,ax2) = plt.subplots(1,2)

#make up linearish data
numberDataPoints=20
x=np.random.uniform(10,30,numberDataPoints)
y=3*x+2+np.random.rand(numberDataPoints)*20

numIterations=20

l1 = simpleLinearFitter(x,y,numIterations)
l1.train()

plt.plot(x,y,'ro')
plt.show()