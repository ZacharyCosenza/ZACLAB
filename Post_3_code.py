# Zachary Cosenza
# Post 3 Code

#%% Part 1 RBF Model Structure

import numpy as np
import matplotlib.pyplot as plt

xc = 0.5 #centerpoint
x = np.linspace(-2,2,100).reshape(1,100)
r = np.sqrt((xc-x)**2)

#Gaussian Model
plt.figure()
sigma = 0.1
phi = np.exp(-(sigma*r)**2)
plt.plot(x,phi,'k.')
plt.xlabel('x')
plt.ylabel('phi')

plt.figure()
plt.plot(r,phi,'r.')
plt.xlabel('r')
plt.ylabel('phi')

#Cubic Model
plt.figure()
phi = r**3
plt.plot(x,phi,'k.')
plt.xlabel('x')
plt.ylabel('phi')

plt.figure()
plt.plot(r,phi,'r.')
plt.xlabel('r')
plt.ylabel('phi')

#Thin Plate Spline
plt.figure()
phi1 = r**2*np.log(r)
plt.plot(x,phi,'k.')
plt.xlabel('x')
plt.ylabel('phi')

plt.figure()
plt.plot(r,phi,'r.')
plt.xlabel('r')
plt.ylabel('phi')

#%% Part 2 Toy Problem and Training Weights

def get_y(x):
    n = len(x)
    y = x**2 + x + 1
    noise = np.random.uniform(low=0,high=y/2,size=(n,1))
    y = y + noise
    return y.reshape(n,1)

x = np.linspace(-2,2,100).reshape(100,1)
yreal = get_y(x)

#Get Small Sample of Y
N = 10
X = np.random.uniform(low = -2,high = 2,size = (N,1))
Y = get_y(X)

#Plot RBF
sigma = .1
params = sigma

def train_RBF(X,Y,params):
    #Get Parameters
    sigma = params
    #Define Centers
    xc = X
    nc = len(xc)
    #Get Weights
    PHI = np.zeros((nc,nc))
    for i in np.arange(nc):
        for j in np.arange(nc):
            r = np.sqrt((X[i]-xc[j])**2)
            PHI[i,j] = np.exp(-(sigma*r)**2)
    w = np.linalg.pinv((PHI.T@PHI))@PHI.T@Y
    return w

w = train_RBF(X,Y,params)

def test_RBF(x,X,params,w,key):
    #Get Parameters
    sigma = params
    #Define Centers
    xc = X
    nc = len(xc)
    n = len(x)
    #Get Weights
    phi = np.zeros((n,nc))
    for i in np.arange(n):
        for j in np.arange(nc):
            r = np.sqrt((x[i]-xc[j])**2)
            phi[i,j] = w[j] * np.exp(-(sigma*r)**2)
    y = np.sum(phi,axis = 1)
    if key == 'reg':
        y = y
    elif key == 'class':
        y = np.sign(y)
    return y.reshape(n,1)

y = test_RBF(x,X,params,w,'reg')

plt.figure()
plt.plot(x,yreal,'k.')
plt.plot(X,Y,'rs')
plt.plot(x,y,'r,')

#%% Part 3 Classification Problem

def get_class(x,y):
    n = len(x)
    c = np.zeros(n)
    for i in np.arange(n):
        if y[i] >= 2:
            c[i] = 1
        else:
            c[i] = -1
    return c

Yclass = get_class(X,Y)
wclass = train_RBF(X,Yclass,params)
yclass = test_RBF(x,X,params,wclass,'class')

plt.figure()
plt.plot(X,Yclass,'rs')
plt.plot(x,yclass,'r.')

yclass_true = get_class(x,get_y(x))
plt.plot(x,yclass_true,'k.')
plt.xlabel('x')
plt.ylabel('class')

#%% Part 4 Polynomial Terms

def train_RBF_polynomial(X,Y,params):
    #Get Parameters
    sigma = params
    #Define Centers
    xc = X
    nc = len(xc)
    #Get Weights
    PHI = np.zeros((nc,nc))
    for i in np.arange(nc):
        for j in np.arange(nc):
            r = np.sqrt((X[i]-xc[j])**2)
            PHI[i,j] = np.exp(-(sigma*r)**2)
    P = np.hstack((np.ones([nc,1]),xc))
    ZEROS = np.zeros([2,2])
    L1 = np.hstack((PHI,P))
    L2 = np.hstack((P.T,ZEROS))
    Lambda = np.vstack((L1,L2))
    gamma = np.vstack((Y,np.zeros([2,1])))
    w = gamma.T @ np.linalg.pinv(Lambda)
    return w

wpoly = train_RBF_polynomial(X,Y,params)

def test_RBF_polynomial(x,X,params,w,key):
    #Get Parameters
    sigma = params
    #Define Centers
    xc = X
    nc = len(xc)
    n = len(x)
    #Get Weights
    phi = np.zeros((n,nc))
    for i in np.arange(n):
        for j in np.arange(nc):
            r = np.sqrt((x[i]-xc[j])**2)
            phi[i,j] = np.exp(-(sigma*r)**2)
    P = np.hstack((np.ones([n,1]),x))
    A = np.hstack((phi,P))
    y = A @ w.T
    if key == 'reg':
        y = y
    elif key == 'class':
        y = np.sign(y)
    return y.reshape(n,1)

ypoly = test_RBF_polynomial(x,X,params,wpoly,'reg')

plt.figure()
plt.plot(x,ypoly,'r,')
plt.plot(x,yreal,'k.')
plt.plot(X,Y,'rs')

#%% Part 5 Hyperparameter Optimization

# Change Sigma
sigma_array = np.linspace(0.1,5,100)
MSE = np.zeros(len(sigma_array))
for i in np.arange(len(sigma_array)):
    sigma = sigma_array[i]
    w = train_RBF(X,Y,sigma)
    y = test_RBF(x,X,sigma,w,'reg')
    MSE[i] = np.sum((y-yreal)**2)
    
plt.figure()
plt.plot(sigma_array,MSE,'g.')
plt.ylabel('MSE')
plt.xlabel('sigma')

#Change Centers
def train_RBF_centers(X,Y,params):
    #Get Parameters
    sigma = params
    num_centers = 3
    #Define Centers
    n = len(X)
    kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(X)
    xc = kmeans.cluster_centers_
    nc = len(xc)
    #Get Weights
    PHI = np.zeros((n,nc))
    for i in np.arange(n):
        for j in np.arange(nc):
            r = np.sqrt((X[i]-xc[j])**2)
            PHI[i,j] = np.exp(-(sigma*r)**2)
    w = np.linalg.pinv((PHI.T@PHI))@PHI.T@Y
    return w,xc

from sklearn.cluster import KMeans

wcenters,xcenters = train_RBF_centers(X,Y,params)

def test_RBF_centers(x,X,params,w,xcenters,key):
    #Get Parameters
    sigma = params
    #Define Centers
    xc = xcenters
    nc = len(xc)
    n = len(x)
    #Get Weights
    phi = np.zeros((n,nc))
    for i in np.arange(n):
        for j in np.arange(nc):
            r = np.sqrt((x[i]-xc[j])**2)
            phi[i,j] = w[j] * np.exp(-(sigma*r)**2)
    y = np.sum(phi,axis = 1)
    if key == 'reg':
        y = y
    elif key == 'class':
        y = np.sign(y)
    return y.reshape(n,1)

y_centers = test_RBF_centers(x,X,params,wcenters,xcenters,'reg')

plt.figure()
plt.plot(x,yreal,'k.')
plt.plot(x,y_centers,'r.')
plt.plot(X,Y,'rs')
plt.plot(xcenters,test_RBF_centers(xcenters,X,params,wcenters,xcenters,'reg'),'bs')