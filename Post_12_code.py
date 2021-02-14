# Zachary Cosenza
# Post 12

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def get_Yfid(X):
    n,p = X.shape
    Y = np.zeros(n)
    for i in np.arange(n):
        if X[i,0] == 0:
            Y[i] = -X[i,1] + np.random.uniform(low = -0.01, high = 0.01)
        else:
            Y[i] = -X[i,1] + np.random.uniform(low = -0.1, high = 0.1)
    return Y

def get_K(x,X,params):
    n,p = x.shape #testing data
    N,p = X.shape #training data
    p = p - 1
    num_IS = int(len(params) / (2 + p))
    K = np.zeros([n,N])
    for i in np.arange(n):
        IS = int(x[i,0])
        for j in np.arange(N):
            #Primary Kernel
            lambdas = params[num_IS*2:num_IS*2+p]
            r = sum((x[i,1:]-X[j,1:])**2 / lambdas**2)
            K[i,j] = params[0]**2 * np.exp(-r/2)
            #Deviation Kernel
            if IS != 0 and IS == X[j,0]:
                lambdas = params[2*num_IS+IS*p:2*num_IS+IS*p+p]
                r = sum((x[i,1:]-X[j,1:])**2 / lambdas**2)
                K[i,j] = K[i,j] + params[IS + num_IS]**2 * np.exp(-r/2)
    return K

#Likelihood Function
def get_like(params,X,Y):
    Y = (Y - Y.mean()) / Y.std()
    N,p = X.shape #training data
    p = p - 1
    ind_IS = np.array(X[:,0],dtype=int)
    s = np.zeros(N)
    for i in range(N):
        s[i] = params[ind_IS[i]]
    S = np.diag(s**2)
    Ky = get_K(X,X,params) + S
    logp = -0.5 * Y.T @ np.linalg.pinv(Ky) @ Y 
    - 0.5 * np.log(np.linalg.det(Ky)) - n/2*np.log(2*np.pi)
    return logp

def get_prior(params,X,Y):
    n,p = X.shape
    p = p - 1
    num_IS = int(len(params) / (2 + p))
    #Prepare Data
    IS0 = (X[:,0] == 0)
    #Initalize Priors
    sigmaf_mean = np.zeros(num_IS)
    lambda_mean = np.ones(p*num_IS) * 1
    for i in np.arange(num_IS):
        IS = (X[:,0] == i)
        if i == 0:
            sigmaf_mean[i] = np.var(Y[IS]) - 0
        else: 
            sigmaf_mean[i] = np.var(Y[IS]) - np.var(Y[IS0])
    sigmaf_sigma = (sigmaf_mean / 2)
    lambda_sigma = (lambda_mean / 2)
    params_prior_mean = np.hstack((sigmaf_mean,lambda_mean))
    params_prior_sigma = np.hstack((sigmaf_sigma,lambda_sigma))
    C_inv = np.linalg.pinv(np.diag(params_prior_sigma))
    mu = params[num_IS:] - params_prior_mean
    prior = -0.5 * mu.T @ C_inv @ mu #log prior of mvn normal
    return prior

def get_Lgrad(params,X,Y):
    n,p = X.shape
    p = p - 1
    ind_IS = np.array(X[:,0],dtype=int)
    num_IS = int(len(params) / (2 + p))
    s = np.zeros(n)
    for i in range(n):
        s[i] = params[ind_IS[i]]
    S = np.diag(s**2)
    K = get_K(X,X,params)
    Ky = K + S
    Lgrad = np.zeros(len(params))
    Ky_inv = np.linalg.pinv(Ky)
    
    for i in np.arange(len(params)):
        if i <= num_IS:
            grad = get_dKdsigma(X,params,i,ind_IS)
        elif i <= 2 * num_IS and i > num_IS:
            grad = get_dKdsigmaf(X,params,ind_IS,i)
        else:
            grad = get_dKdlam(X,params,i,K)
        A = 0.5 * Y.T @ Ky_inv @ grad @ Ky_inv @ Y - 0.5 * np.trace(Ky_inv @ grad)
        Lgrad[i] = A
    return np.array(Lgrad)

def get_dKdsigma(X,params,i,ind_IS):
    n,p = X.shape
    v = np.zeros(n)
    v[ind_IS] = 2 * params[i]
    dKdsigma = np.diag(v) 
    return dKdsigma

def get_k(x,X,params,param_IS):
    p = len(x)
    p = p - 1
    num_IS = int(len(params) / (2 + p))
    lambdas = params[2*num_IS+param_IS*p:2*num_IS+param_IS*p+p]
    r = sum((x[1:]-X[1:])**2 / lambdas**2)
    k = params[param_IS + num_IS]**2 * np.exp(-r/2)
    return k

def get_dKdsigmaf(X,params,ind_IS,k):
    n,p = X.shape
    p = p - 1
    num_IS = int(len(params) / (2 + p))
    param_IS = k - num_IS - 1
    dKdsigmaf = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            if param_IS == 0:
                dKdsigmaf[i,j] = get_k(X[i,:],X[j,:],params,param_IS) * 2 / params[num_IS + param_IS]
            else:
                if X[i,0] == param_IS and X[j,0] == param_IS:
                    dKdsigmaf[i,j] = get_k(X[i,:],X[j,:],params,param_IS) * 2 / params[num_IS + param_IS]
    return dKdsigmaf

def get_dKdlam(X,params,k,K):
    n,p = X.shape
    p = p - 1
    num_IS = int(len(params) / (2 + p))
    # TEMPORARY!!!
    if k - 2 * num_IS > 0 and k - 2 * num_IS < p:
        param_IS = 0
    else:
        param_IS = 1
    param_ind = k - 2 * num_IS - p * param_IS - 1
    dKdlam = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            r = (X[i,param_ind+1] - X[j,param_ind]+1)**2 * params[k]**(-3)
            if param_IS == 0:   
                dKdlam[i,j] =  r * get_k(X[i,:],X[j,:],params,param_IS)
            else:
                if X[i,0] == param_IS and X[j,0] == param_IS:
                    dKdlam[i,j] = r * get_k(X[i,:],X[j,:],params,param_IS)
    return dKdlam

def get_mu(x,X,Y,params):
    # Y = (Y - Y.mean()) / Y.std()
    n,p = x.shape #testing data
    N,p = X.shape #training data
    p = p - 1
    ind_IS = np.array(X[:,0],dtype=int)
    s = np.zeros(N)
    y = np.zeros(n)
    sigma2 = np.zeros(n)
    for i in range(N):
        s[i] = params[ind_IS[i]]
    S = np.diag(s**2)
    Ky_inv = np.linalg.pinv(get_K(X,X,params) + S)
    for i in range(n):
        xin = x[i,].reshape(1,p+1)
        KxX = get_K(xin,X,params)
        Kxx = get_K(xin,xin,params)
        y[i] = KxX @ Ky_inv @ Y
        sigma2[i] = Kxx - KxX @ Ky_inv @ KxX.T
    # y = y * Y.std() + Y.mean()
    return y,sigma2

#Make Wrappers of BFGS for scipy optimize minimize
def wrapper_L(p,X,Y):
    return -1 * get_like(p,X,Y) #neg bc maximizing

def wrapper_dL(p,X,Y):
    return -1 * get_Lgrad(p,X,Y) #neg bc maximizing

#%% Regular GP Model

N = 10
p = 2
n = 1000
X = np.random.uniform(low = 0, high = 1, size = (N,p+1))
X[:,1] = np.linspace(0,1,N)
x0 = np.random.uniform(low = 0, high = 1, size = (n,p+1))
x1 = np.random.uniform(low = 0, high = 1, size = (n,p+1))
x0[:,1] = np.linspace(0,1,n)
x1[:,1] = np.linspace(0,1,n)
X[:,0] = 0
x0[:,0] = 0
x1[:,0] = 1
X[:6,0] = 1
X = np.repeat(X,3,axis = 0)
Y = get_Yfid(X)
y0 = get_Yfid(x0)
y1 = get_Yfid(x1)
r0 = X[:,0] == 0
r1 = X[:,0] == 1

#Define Hyperparameters
#params = [sigma, sigmaf, lambda]
num_IS = 2
num_hypers = (2 + p) * num_IS
param_low = 0.001
param_high = 3
param_bounds = []
for i in range(num_hypers):
    bound = (param_low,param_high)
    param_bounds.append(bound)

# num_samples = 1
# like = np.zeros(num_samples)
# params_list = np.random.uniform(low = param_low, high = param_high, 
#                                 size = (num_samples,num_hypers))
# for i in range(num_samples):
#     like[i] = get_like(params_list[i,:],X,Y)
# ind_sample = np.argmax(like)
# params = params_list[ind_sample,:]

params_0 = np.random.uniform(low = param_low, high = param_high, size = num_hypers)
bfgs_res = minimize(wrapper_L,x0 = params_0,args=(X,Y), method='L-BFGS-B', 
               jac = wrapper_dL, bounds = param_bounds)
params = bfgs_res['x']
print(bfgs_res)

mu,sigma2 = get_mu(x0,X,Y,params)
MU,_ = get_mu(X,X,Y,params)

# R2 Plot
plt.figure()
plt.plot(Y[r0],Y[r0],'bs') #plot 0th true data
plt.plot(y0,mu,'r.') #plot 0th prediction
plt.plot(y0,y0,'b.') #plot true process
plt.xlabel('y true')
plt.ylabel('y pred')

# Example Plot
plt.figure()
plt.plot(x0[:,1],y0,'b.') #plot 0th true process
plt.plot(x1[:,1],y1,'g.') #plot 1th true process
plt.plot(X[r0,1],Y[r0],'bs') #plot 0th data
plt.plot(X[r1,1],Y[r1],'gs') #plot 1st data
plt.plot(x0[:,1],mu,'r.') #plot 0th prediction
plt.xlabel('x')
plt.ylabel('y')

#%% GP for Multifidelity Model using Botorch

import torch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.models import SingleTaskGP

#Convert from Numpy to Torch
X = torch.tensor(X)
Y = torch.tensor(Y).unsqueeze(-1)

model_fid = SingleTaskMultiFidelityGP(X,Y,data_fidelity=0,outcome_transform=Standardize(m=1))
mll_fid = ExactMarginalLogLikelihood(model_fid.likelihood,model_fid)
fit_gpytorch_model(mll_fid)

model = SingleTaskGP(X[r0,1:],Y[r0],outcome_transform=Standardize(m=1))
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)

with torch.no_grad():
    y_bo_fid = model_fid.posterior(torch.from_numpy(x0)).mean
    # std_bo_fid = model_fid.posterior(torch.from_numpy(x)).variance
    y_bo = model.posterior(torch.from_numpy(x0[:,1:])).mean

plt.figure()
plt.plot(Y[r0],Y[r0],'rs')
plt.plot(y0,y_bo_fid,'r.')
# plt.plot(y,y_bo_fid+np.sqrt(std_bo_fid),'r.')
# plt.plot(y,y_bo_fid-np.sqrt(std_bo_fid),'r.')
plt.plot(y0,y_bo,'b')
plt.plot(y0,y0,'k.')
plt.xlabel('y true')
plt.ylabel('y pred')

plt.figure()
plt.plot(x0[:,1],y0,'k.')
plt.plot(X[r0,1],Y[r0],'rs')
plt.plot(X[r1,1],Y[r1],'gs')
plt.plot(x0[:,1],y_bo_fid,'r.')
plt.plot(x0[:,1],y_bo,'b.')
plt.xlabel('x_1')
plt.ylabel('y')