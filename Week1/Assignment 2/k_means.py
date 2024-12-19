import numpy as np
import matplotlib.pyplot as plt
import time
import os

# Load data from a CSV (comma-separated) file
data = np.loadtxt('data.txt', delimiter=',')

data_path = "data.txt"


### TODO 2
### Load data from data_path
### Check the input file spice_locations.txt to understand the Data Format
### Return : np array of size Nx2
def load_data(data_path):
    return np.loadtxt(data_path, delimiter=',')


### TODO 3.1
### If init_centers is None, initialize the centers by selecting K data points at random without replacement
### Else, use the centers provided in init_centers
### Return : np array of size Kx2
def initialise_centers(data, K, init_centers=None):
    if init_centers==None:
        indices = np.random.choice(data.shape[0],size=K,replace=False)
        init_centers = data[indices]
        return init_centers
    init_centers=init_centers[0:K,:]
    return init_centers
### TODO 3.2
### Initialize the labels to all ones to size (N,) where N is the number of data points
### Return : np array of size N
def initialise_labels(data):
    return np.ones(data.shape[0])

### TODO 4.1 : E step
### For Each data point, find the distance to each center
### Return : np array of size NxK
def calculate_distances(data, centers):
    #returns 2d euclid dist
    def dist(pt1,pt2):
        return np.sqrt((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)
    distances = np.zeros((data.shape[0],centers.shape[0]))
    for i in range(data.shape[0]):
        for j in range(centers.shape[0]):
            distances[i,j]=dist(data[i],centers[j])
    return distances

### TODO 4.2 : E step
### For Each data point, assign the label of the nearest center
### Return : np array of size N
def update_labels(distances):
    return np.argmin(distances,axis=1)

### TODO 5 : M step
### Update the centers to the mean of the data points assigned to it
### Return : np array of size Kx2
def update_centers(data, labels, K):
    new_centers =np.array([np.mean(data[labels==i], axis=0 ) for i in range(K)])
    return new_centers

### TODO 6 : Check convergence
### Check if the labels have changed from the previous iteration
### Return : True / False
def check_termination(labels1, labels2):
    return labels1==labels2

### simulate the algorithm in the following function. run.py will call this
### function with given inputs.
def kmeans(data_path:str, K:int, init_centers):
    '''
    Input :
        data (type str): path to the file containing the data
        K (type int): number of clusters
        init_centers (type numpy.ndarray): initial centers. shape = (K, 2) or None
    Output :
        centers (type numpy.ndarray): final centers. shape = (K, 2)
        labels (type numpy.ndarray): label of each data point. shape = (N,)
        time (type float): time taken by the algorithm to converge in seconds
    N is the number of data points each of shape (2,)
    '''
    data = load_data(data_path)
    centers = initialise_centers(data, K, init_centers)
    labels0 = initialise_labels(data)
    labels1=labels0+1
    start_time = time.time()
    while (labels0!=labels1).all():
        labels0=labels1
        distances = calculate_distances(data, centers)
        labels1 = update_labels(distances)
        centers = update_centers(data, labels1, K)
        visualise(data_path,labels1,centers)
    end_time = time.time()
    return (centers,labels1,end_time-start_time)

### to visualise the final data points and centers.
def visualise(data_path, labels, centers):
    data = load_data(data_path)

    # Scatter plot of the data points
    plt.scatter(data[:, 0], data[:, 1], c=labels, s=50, cmap='viridis')
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)

    return plt
x,y,*_ = kmeans(data_path, 5, None)
visualise(data_path, y,x)
plt.show()