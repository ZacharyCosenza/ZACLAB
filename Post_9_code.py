# Zachary Cosenza
# Post 9

#%% Basic SGD on Least Squares Problem

import numpy as np
import random
import matplotlib.pyplot as plt

#Define Problem
N,p = 10**3, 10**2
X = np.random.uniform(low = 0, high = 1, size = (N,p))
Y = 1 * X[:,0] + np.random.uniform(low = -0.2, high = 0.2, size = N)

#Regular OLS
beta_OLS = np.linalg.pinv(X.T @ X) @ X.T @ Y

#Regular SGD

def get_like(X,Y,beta):
    like = sum((beta @ X.T - Y)**2)
    return like

def get_like_grad(X,Y,beta):
    n,p = X.shape
    like_grad = np.zeros(p)
    for i in range(p):
        like_grad[i] = sum(2 * (beta @ X.T - Y) * X[:,i])
    return like_grad / n

iter_max = 1000
B = 100
eta = 0.02
beta = np.random.uniform(low = -1,high = 1,size = p)
f = np.zeros(iter_max)
for i in range(iter_max):
    ind_batch = random.sample(range(N),B)
    X_batch = X[ind_batch,:]
    Y_batch = Y[ind_batch]
    g = get_like_grad(X_batch,Y_batch,beta)
    beta = beta - eta * g
    f[i] = np.log(get_like(X,Y,beta))
    
plt.figure(1)
plt.plot(np.arange(iter_max),f,'k')
plt.xlabel('Iteration')
plt.ylabel('Log of MSE')

plt.figure(2)
plt.plot(Y,Y,'k.')
plt.plot(Y,beta @ X.T,'r.')
plt.plot(Y,beta_OLS @ X.T,'b.')
plt.xlabel('Y True')
plt.ylabel('Y Predicted')

#%% Part 2 SGD with Random Variables

#Define Problem
N,p = 10**2, 10**1
X = np.random.uniform(low = 0, high = 1, size = (N,p))
Y = 1 * X[:,0] + np.random.uniform(low = -0.2, high = 0.2, size = N)

#Regular OLS
beta_OLS = np.linalg.pinv(X.T @ X) @ X.T @ Y

#SGD Attempt

sigma = 0.25

def get_stoch_grad(X,Y,sigma,B,beta):
    N,p = X.shape
    stoch_grads = np.zeros([p,B])
    for b in range(B):
        beta_sample = np.random.multivariate_normal(beta,np.eye(p)*sigma)
        stoch_grads[:,b] = get_like_grad(X,Y,beta_sample)
    stoch_grad = np.mean(stoch_grads, axis = 1)
    return stoch_grad
    
iter_max = 1000
B = 10
eta = 0.02
beta = np.random.uniform(low = -1,high = 1,size = p)
f = np.zeros(iter_max)
for i in range(iter_max):
    g = get_stoch_grad(X,Y,sigma,B,beta)
    beta = beta - eta * g
    f[i] = np.log(get_like(X,Y,beta))
    
plt.figure(3)
plt.plot(np.arange(iter_max),f,'k')
plt.xlabel('Iteration')
plt.ylabel('Log of MSE')

plt.figure(4)
plt.plot(Y,Y,'k.')
plt.plot(Y,beta @ X.T,'r.')
plt.plot(Y,beta_OLS @ X.T,'b.')
plt.xlabel('Y True')
plt.ylabel('Y Predicted')

#%% Part 3 Advanced SGD ADAGRAD

#Define Problem
N,p = 10**2, 10
X = np.random.uniform(low = 0, high = 1, size = (N,p))
Y = 1 * X[:,0] + np.random.uniform(low = -0.2, high = 0.2, size = N)

#Regular OLS
beta_OLS = np.linalg.pinv(X.T @ X) @ X.T @ Y

iter_max = 1000
B = 100
eta = 0.03
beta = np.random.uniform(low = -1,high = 1,size = p)
f = np.zeros(iter_max)
G = np.zeros(p)
for i in range(iter_max):
    ind_batch = random.sample(range(N),B)
    X_batch = X[ind_batch,:]
    Y_batch = Y[ind_batch]
    g = get_like_grad(X_batch,Y_batch,beta)
    G = G + g**2
    beta = beta - eta * g / np.sqrt(G)
    f[i] = np.log(get_like(X,Y,beta))

plt.figure(5)
plt.plot(np.arange(iter_max),f,'k')
plt.xlabel('Iteration')
plt.ylabel('Log of MSE')

plt.figure(6)
plt.plot(Y,Y,'k.')
plt.plot(Y,beta @ X.T,'r.')
plt.plot(Y,beta_OLS @ X.T,'b.')
plt.xlabel('Y True')
plt.ylabel('Y Predicted')

#%% Part 4 Advanced SGD RMSPROP

#Define Problem
N,p = 10**2, 10
X = np.random.uniform(low = 0, high = 1, size = (N,p))
Y = 1 * X[:,0] + np.random.uniform(low = -0.2, high = 0.2, size = N)

#Regular OLS
beta_OLS = np.linalg.pinv(X.T @ X) @ X.T @ Y

iter_max = 1000
B = 100
eta = 0.03
gamma = 0.2
beta = np.random.uniform(low = -1,high = 1,size = p)
f = np.zeros(iter_max)
v = np.zeros(p)
for i in range(iter_max):
    ind_batch = random.sample(range(N),B)
    X_batch = X[ind_batch,:]
    Y_batch = Y[ind_batch]
    g = get_like_grad(X_batch,Y_batch,beta)
    v = gamma * v + (1 - gamma) * (g)**2
    beta = beta - eta * g / np.sqrt(v)
    f[i] = np.log(get_like(X,Y,beta))

plt.figure(7)
plt.plot(np.arange(iter_max),f,'k')
plt.xlabel('Iteration')
plt.ylabel('Log of MSE')

plt.figure(8)
plt.plot(Y,Y,'k.')
plt.plot(Y,beta @ X.T,'r.')
plt.plot(Y,beta_OLS @ X.T,'b.')
plt.xlabel('Y True')
plt.ylabel('Y Predicted')

#%% Part 5 Advanced SGD ADAM

#Define Problem
N,p = 10**2, 10
X = np.random.uniform(low = 0, high = 1, size = (N,p))
Y = 1 * X[:,0] + np.random.uniform(low = -0.2, high = 0.2, size = N)

#Regular OLS
beta_OLS = np.linalg.pinv(X.T @ X) @ X.T @ Y

def get_dL(X,Y,beta_new,beta_old):
    like_old = get_like(X,Y,beta_old)
    like_new = get_like(X,Y,beta_new)
    dlike = like_new - like_old
    dbeta = beta_new - beta_old
    if any(dbeta) == 0:
        dL = 0
    else:
        dL = dlike / dbeta
    return dL

iter_max = 100
B = 100
eta = 1
e = 10^-8
m = np.zeros(p)
v = np.zeros(p)
beta1 = 0.5
beta2 = 0.5
beta_old = np.random.uniform(low = -1,high = 1,size = p)
f = np.zeros(iter_max)
for i in range(iter_max):
    ind_batch = random.sample(range(N),B)
    X_batch = X[ind_batch,:]
    Y_batch = Y[ind_batch]
    g = get_like_grad(X_batch,Y_batch,beta_old)
    m = beta1*m + (1-beta1)*g
    v = beta2*v + (1-beta2)*(g)**2
    m_hat = m / (1-beta1**(1+i))
    v_hat = v / (1-beta2**(1+i))
    beta_new = beta_old + eta * m_hat / (e + np.sqrt(v_hat))
    f[i] = np.log(get_like(X,Y,beta_new))
    beta_old = beta_new

plt.figure(9)
plt.plot(np.arange(iter_max),f,'k')
plt.xlabel('Iteration')
plt.ylabel('Log of MSE')

plt.figure(10)
plt.plot(Y,Y,'k.')
plt.plot(Y,beta_new @ X.T,'r.')
plt.plot(Y,beta_OLS @ X.T,'b.')
plt.xlabel('Y True')
plt.ylabel('Y Predicted')