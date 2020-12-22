# Zachary Cosenza
# Post 2 Code

#%% Part 1 Gaussian Processes

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

#Actual Function
def get_Y(X):
    n,p = X.shape
    beta = np.array([1,-1])
    Y = (X @ beta).reshape(-1,1) + np.random.uniform(low = 0, high = 0.1,size = (n,1))
    return Y

#Covariance Kernel
def get_K(x,X,params):
    sigmaf = params[1]
    lam = params[2:]
    n,p = x.shape #test data
    N,p = X.shape #training data
    K = np.zeros([n,N])
    for i in np.arange(n):
        for j in np.arange(N):
            r = (x[i,:]-X[j,:]).T @ (x[i,:]-X[j,:])
            # r = np.sum((x[i,:] - X[j,:])**2)
            K[i,j] = sigmaf**2 * np.exp(-0.5*r/lam**2)
    return K

#Mean GP
def get_mean(x,X,Y,params):
    sigma = params[0]
    n,p = X.shape #training data
    KxX = get_K(x,X,params)
    KXX = get_K(X,X,params)
    y = KxX @ np.linalg.pinv(KXX + sigma**2*np.eye(n)) @ Y
    return y

#STD Squared of GP
def get_std_squared(x,X,params):
    Kxx = get_K(x,x,params)
    KxX = get_K(x,X,params)
    KXX = get_K(X,X,params)
    KXx = get_K(X,x,params)
    std_squared = Kxx - KxX @ np.linalg.pinv(KXX) @ KXx
    return std_squared

def PlotGPContour(X,Y,params,key):
    l = 20
    x1 = np.linspace(0,1,l)
    x2 = x1
    C = np.zeros([l,l])
    for i in np.arange(l):
        for j in np.arange(l):
            x = np.array([x1[i],x2[j]]).reshape(1,2)
            if key == 'mean':
                C[i,j] = get_mean(x,X,Y,params)
            elif key == 'EI':
                C[i,j] = get_EI(x,X,Y,params)
            elif key == 'std':
                C[i,j] = np.sqrt(np.abs(get_std_squared(x,X,params)))
            elif key == 'bo':
                with torch.no_grad():
                    x = np.array([x1[i],x2[j]]).reshape(1,2)
                    bo = params.posterior(torch.from_numpy(x))
                    C[i,j] = bo.mean.cpu().numpy()
    return C

def MakePlot(X,Y,params,key):
    l = 20
    plt.figure(figsize=(5,3))
    xx, yy = np.meshgrid(np.linspace(0, 1, l), np.linspace(0, 1, l), sparse=True)
    C = PlotGPContour(X,Y,params,key)
    plt.contourf(np.linspace(0, 1, l),np.linspace(0, 1, l),C)
    plt.plot(X[:,1],X[:,0],'k+')
    plt.xlabel('x_2')
    plt.ylabel('x_1')
    plt.colorbar()

#Training Data
N = 8
X = np.random.uniform(size = (N,2))
Y = get_Y(X)

l = 20
plt.figure(figsize=(5,3))
xx, yy = np.meshgrid(np.linspace(0, 1, l), np.linspace(0, 1, l), sparse=True)
beta = np.array([1,-1])
z = beta[0] * xx + beta[1] * yy
plt.contourf(np.linspace(0, 1, l),np.linspace(0, 1, l),z.T)
plt.plot(X[:,0],X[:,1],'k+')
plt.xlabel('x_2')
plt.ylabel('x_1')
plt.colorbar()

#Testing Data
n = 1000
x = np.random.uniform(size = (n,2))
y = get_Y(x)

sigma = 1
sigmaf = 1
lam = 1
params = np.array([sigma,sigmaf,lam])

#%% Part 2 Hyperparameter Optimization

#Likelihood Function
def get_like(params,X,Y):
    n,p = X.shape
    sigma = params[0]
    Ky = get_K(X,X,params) + sigma**2 * np.eye(n)
    logp = -0.5 * Y.T @ np.linalg.pinv(Ky) @ Y - 0.5 * np.log(np.linalg.det(Ky)) - n/2*np.log(2*np.pi)
    return logp

#Gradient of Likelihood Function
def get_Lgrad(params,X,Y):
    n,p = X.shape
    sigma = params[0]
    Ky = get_K(X,X,params) + sigma**2 * np.eye(n)
    Lgrad = np.zeros(len(params))
    Ky_inv = np.linalg.pinv(Ky)
    
    for i in np.arange(3):
        if i == 0:
            grad = get_dKdsigma(X,params)
        elif i == 1:
            grad = get_dKdsigmaf(X,params)
        elif i == 2:
            grad = get_dKdlam(X,params)
        A = 0.5 * Y.T @ Ky_inv @ grad @ Ky_inv @ Y - 0.5 * np.trace(Ky_inv @ grad)
        Lgrad[i] = A
    return np.array(Lgrad)

def get_dKdsigma(X,params):
    sigma = params[0]
    n,p = X.shape
    dKdsigma = 2 * sigma * np.eye(n)
    return dKdsigma

def get_dKdsigmaf(X,params):
    sigmaf = params[1]
    K = get_K(X,X,params)
    dKdsigmaf = K * 2 / sigmaf
    return dKdsigmaf

def get_dKdlam(X,params):
    lam = params[2]
    K = get_K(X,X,params)
    n,p = X.shape
    dKdlam = np.zeros([n,n])
    for i in np.arange(n):
        for j in np.arange(n):
            r = (X[i,:]-X[j,:]).T @ (X[i,:]-X[j,:])
            dKdlam[i,j] = K[i,j] * r / lam**3
    return dKdlam

#Plot some Parameters
sigma_list = np.linspace(0.1,10,10)
sigmaf_list = np.linspace(0.1,10,10)
lam_list = np.linspace(0.1,10,10)
like_sigma = [get_like(np.array([sigma,20,10]),X,Y) for sigma in sigma_list]
like_sigmaf = [get_like(np.array([0.3,sigmaf,10]),X,Y) for sigmaf in sigmaf_list]
like_lam = [get_like(np.array([0.3,20,lam]),X,Y) for lam in lam_list]

plt.figure()
plt.subplot(1,3,1)
plt.plot(sigma_list,np.array(like_sigma).reshape(10,1))
plt.xlabel('sigma')
plt.ylabel('likelihood')
plt.subplot(1,3,2)
plt.plot(sigmaf_list,np.array(like_sigmaf).reshape(10,1))
plt.xlabel('sigmaf')
plt.subplot(1,3,3)
plt.plot(lam_list,np.array(like_lam).reshape(10,1))
plt.xlabel('lambda')

# Random Search for Optimal Hyperparameteres
num_rand = 100
params_rand = np.random.uniform(low = 0.1, high = 20, size = (num_rand,3))
like_rand = []
for i in np.arange(num_rand):
    like_rand.append(get_like(params_rand[i,:],X,Y))
ind_rand = np.argmax(like_rand)
params_rand_opt = params_rand[ind_rand,:]

#Plot Contours
MakePlot(X,Y,params_rand_opt,'mean')

# Gradient Descent Method

params = np.array([10,10,10])
iter_max = 1000
eta = 0.1
Lplot = []
params_gd = np.zeros([iter_max,3])

for i in np.arange(iter_max):
    Lgrad = get_Lgrad(params,X,Y)
    params = params + eta * Lgrad
    projection = np.argwhere(params < 0)
    params[projection] = 0.1
    Lplot.append(get_like(params,X,Y))
    params_gd[i,:] = params
params_gd_opt = params
plt.figure()
plt.plot(np.arange(len(Lplot)).reshape(len(Lplot),1),np.array(Lplot).reshape(len(Lplot),1))
plt.xlabel('Iteration')
plt.ylabel('likelihood')

#Make Plot to Show Gradient Descent
l = 20
param1_list = np.linspace(params_gd[:,0].min(),params_gd[:,0].max(),l)
param2_list = np.linspace(params_gd[:,1].min(),params_gd[:,1].max(),l)
param3 = 0.1
C = np.zeros([l,l])
i = 0
for param1 in param1_list:
    j = 0
    for param2 in param2_list:
        params = np.array([param1,param2,param3])
        C[i,j] = get_like(params,X,Y)
        j = j + 1
    i = i + 1
plt.figure(figsize=(5,3))
plt.contourf(param1_list,param2_list,C)
plt.xlabel('param2 = sigma')
plt.ylabel('param1 = sigmaf')
plt.plot(params_gd[:,0],params_gd[:,1],'r.')
plt.colorbar()

#Plot Contours
MakePlot(X,Y,params_gd_opt,'mean')

# BFGS Method

#Make Wrappers of BFGS for scipy optimize minimize
def wrapper_L(p,X,Y):
    return -1 * get_like(p,X,Y) #neg bc maximizing

def wrapper_dL(p,X,Y):
    return -1 * get_Lgrad(p,X,Y) #neg bc maximizing

params_0 = np.array([10,10,10])
bfgs_res = minimize(wrapper_L,x0 = params_0,args=(X,Y), method='L-BFGS-B', 
               jac = wrapper_dL, bounds = ((0.1,20), (0.1,20), (0.1,20)))
params_bfgs_opt = bfgs_res['x']

#Plot Contours
MakePlot(X,Y,params_bfgs_opt,'mean')

#%% Part 3 Expected Improvement Function

def get_EI(x,X,Y,params):
    n,p = x.shape
    std = np.zeros([n,1])
    for i in np.arange(n):
        std[i] = np.sqrt(np.abs(get_std_squared(x[i,:].reshape(1,2),X,params)))
    y = get_mean(x,X,Y,params)
    Y_min = Y.min()
    z = (Y_min - y) / std
    EI = (Y_min - y) * norm.cdf(z) + std * norm.pdf(z)
    EI[std == 0] = 0
    return EI

MakePlot(X,Y,params_bfgs_opt,'std') #Plot Standard Deviation
MakePlot(X,Y,params_bfgs_opt,'EI') #Plot EI Contour

#%% Part 4 Botorch

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")# use a GPU if available
dtype = torch.float
train_X = torch.from_numpy(X)
train_Y = torch.from_numpy(Y)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)
model.eval() # set model (and likelihood)
    
MakePlot(X,Y,model,'bo') #Plot Contour

#%% Comparison of Models

#Testing Data
N = 1000
xtest = np.random.uniform(size = (N,2))
ytest = get_Y(xtest)

# My Model
yrand = get_mean(xtest,X,Y,params_rand_opt)
ygd = get_mean(xtest,X,Y,params_gd_opt)
ybfgs = get_mean(xtest,X,Y,params_bfgs_opt)

# Botorch Model
with torch.no_grad():
    xtest = torch.from_numpy(xtest)
    bo = model.posterior(xtest)
    ybo = bo.mean.cpu().numpy()

error_rand = sum((ytest-yrand)**2)
error_gd = sum((ytest-ygd)**2)
error_bfgs = sum((ytest-ybfgs)**2)
error_bo = sum((ybo-ytest)**2)
print(error_rand)
print(error_gd)
print(error_bfgs)
print(error_bo)

plt.figure()
plt.plot(ytest,yrand,'r.')
plt.plot(ytest,ygd,'g.')
plt.plot(ytest,ybfgs,'y.')
plt.plot(ytest,ybo,'b.')
plt.plot(ytest,ytest,'k.')
plt.xlabel('True Y')
plt.ylabel('Predicted Y')