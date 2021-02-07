# Zachary Cosenza
# Post 11

#%% Part 1 Basic of GA

import numpy as np
import matplotlib.pyplot as plt

def toy_problem(x):
    y = x[0]**2 - x[0] * 0.5 + x[1] + 1
    y = y + np.random.uniform(low = -0.2, high = 0.2)
    return y

def get_GA(funct,p,*args):
    #Inital Problem
    iter_max = 100
    N = 10
    x = np.random.uniform(low = 0, high = 1, size = (N,p))
    y = np.zeros(N)
    for i in range(N):
        y[i] = funct(x[i,:],*args)
    X_list = []
    Y_list = []

    for k in range(iter_max):  
        print(k)
        # Selection of Best Points (Roulette Wheel Selection)
        P_max = y / sum(y)
        P_min = 1 - P_max
        P_min = P_min / sum(P_min)
        l = int(len(y)/2)
        #Minimization
        ind = np.random.choice(a = np.arange(len(y)),
                                    size = l,p = P_min, replace = False)
        x_parents = x[ind,:]
        
        # Crossover of Best Points (k-Point)
        num_children = len(y) - l
        x_children = np.zeros([num_children,p])
        for i in range(num_children):
            #Get Two Parents
            parents = np.random.choice(l, 2, replace = False)
            x_couple = x_parents[parents,:]
            #Cut Point
            cut = np.random.randint(low = 0, high = len(y))
            #Crossover
            x_children[i,:] = np.concatenate((x_couple[0,0:cut],x_couple[1,cut:p]))
        
        #Combination
        x = np.concatenate((x_parents,x_children))
        
        #Mutation
        p_mut = 0.01
        for i in range(N):
            for j in range(p):
                r = np.random.uniform(low = 0, high = 1)
                if r <= p_mut:
                    x[i,j] = np.random.uniform(low = 0, high = 1)
        y = np.zeros(N)
        for i in range(N):
            y[i] = funct(x[i,:],*args)
        X_list.append(x)
        Y_list.append(y)
    return X_list,Y_list

X,Y = get_GA(toy_problem,10)

#Plot Everything
iterations = np.arange(len(Y))
y_best = np.zeros(len(Y))
y_best_overall = np.zeros(len(Y))
for i in range(len(Y)):
    y_best[i] = min(Y[i])
    if i == 0:
        y_best_overall[i] = y_best[i]
    else:
        if y_best[i] < min(y_best[:i]):
            y_best_overall[i] = y_best[i]
        else:
            y_best_overall[i] = y_best_overall[i-1]
plt.plot(iterations,y_best)
plt.plot(iterations,y_best_overall)
plt.xlabel('Iteration of GA')
plt.legend(['Best Point','Cumulative Best Point'])

#%% Part 2 Example Problem using LBFGSB

import scipy.optimize

def get_y(X): #real
    n,p = X.shape
    Y = np.zeros(n)
    for i in range(n):
        if X[i,0] == 0:
            Y[i] = X[i,1]
        elif X[i,0] == 1:
            Y[i] = 2 * X[i,1] + 1
        else:
            Y[i] = 1 * X[i,1] + np.random.uniform(low = -0.5, high = 0.5)
    return Y

def get_K(x,X,params,num_IS):
    n,p = x.shape #test data
    N,p = X.shape #training data
    p = p - 1 #account for label in x
    K = np.zeros([n,N])
    for i in range(n):
        IS0 = 0
        IS = int(x[i,0])
        for j in range(N):
            C_inv_0 = np.linalg.pinv(np.diag(params[(p+2)*IS0:(p+2)*IS0+p]**2))
            r = (x[i,1:]-X[j,1:]).T  @ C_inv_0 @ (x[i,1:]-X[j,1:])
            #Primary Kernel
            K[i,j] = params[IS0+num_IS]**2 * np.exp(-r/2)
            #Deviation Kernel
            if IS == X[j,0] and IS != IS0:
                C_inv = np.linalg.pinv(np.diag(params[(p+2)*IS:(p+2)*IS+p]**2))
                r = (x[i,1:]-X[j,1:]).T  @ C_inv @ (x[i,1:]-X[j,1:])
                K[i,j] = K[i,j] + params[IS+num_IS]**2 * np.exp(-r/2)
            if i == j:
                K[i,j] = K[i,j] + params[IS]
    return K

def get_sigma2(x,X,params,num_IS):
    n,p = x.shape
    N,_ = X.shape
    sigma2 = np.zeros(n)
    KXX = get_K(X,X,params,num_IS)
    for i in range(n):
        xin = x[i,:].reshape(1,p)
        Kxx = get_K(xin,xin,params,num_IS)
        KxX = get_K(xin,X,params,num_IS)
        sigma2[i] = Kxx - KxX @ np.linalg.pinv(KXX) @ KxX.T
    return sigma2

#Likelihood Function
def get_like(params,X,Y,num_IS):
    n,p = X.shape
    p = p - 1
    Ky = get_K(X,X,params,num_IS)
    logp = -0.5 * Y.T @ np.linalg.pinv(Ky) @ Y 
    - 0.5 * np.log(np.linalg.det(Ky)) - n/2*np.log(2*np.pi)
    return -logp

#Likelihood Function w/ Priors
def get_like_priors(params,X,Y,num_IS):
    n,p = X.shape
    p = p - 1
    Ky = get_K(X,X,params,num_IS)
    logp = -0.5 * Y.T @ np.linalg.pinv(Ky) @ Y 
    - 0.5 * np.log(np.linalg.det(Ky)) - n/2*np.log(2*np.pi) 
    + get_prior(params,X,Y,num_IS)
    return -logp

def get_prior(params,X,Y,num_IS):
    n,p = X.shape
    p = p - 1
    #Prepare Data
    IS0 = (X[:,0] == 0)
    #Initalize Priors
    sigmaf_mean = np.zeros(num_IS)
    lambda_mean = np.ones(p*num_IS) * 1
    for i in range(num_IS):
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

#Define Problem Training Data
num_IS = 3
N1, N2, N3, p = 10,5,5,1
N = N1 + N2 + N3
X1 = np.zeros([N1,p+1])
X2 = np.zeros([N2,p+1])
X3 = np.zeros([N3,p+1])
X1[:,1] = np.random.uniform(low = 0, high = 0.5, size = N1)
X2[:,1] = np.random.uniform(low = 0.5, high = 1, size = N2)
X3[:,1] = np.random.uniform(low = 0.5, high = 1, size = N3)
X1[:,0] = 0
X2[:,0] = 1
X3[:,0] = 2
X = np.concatenate((X1,X2,X3))

# Replicates
rep = 3
X = np.repeat(X,rep,axis = 0)
N1, N2, N3, N = rep * N1, rep * N2, rep * N3, rep * N
Y = get_y(X)

#Get Testing Data
n = 100
x = np.zeros([n,p+1])
x[:,1] = np.linspace(0,1,n)

#Define Hyperparameters
# params = [sigmas ... sigmafs ... lambdas]
# lambdas = [lambda_IS0_p0 lambda_IS0_p1 ...]
num_hypers = num_IS * 2 + num_IS * p
param = np.ones(num_hypers)

#Define Hyperparameter Bounds
sigma_low, sigmaf_low, lambda_low = 1e-2, 1e-2, 1e-2
sigma_high, sigmaf_high, lambda_high = 10, 10, 10

param_bounds = []
for i in range(len(param)):
    bound = (sigma_low,sigma_high)
    param_bounds.append(bound)

bfgs_info = scipy.optimize.fmin_l_bfgs_b(func = get_like_priors, 
                             x0 = param, args=(X,Y,num_IS), 
                             approx_grad=True,
                             bounds = param_bounds, 
                             m=10, 
                             factr=10.0, 
                             pgtol=0.01,
                             epsilon=1e-08, 
                             iprint=-1, 
                             maxfun=15000, 
                             maxiter=100, 
                             disp=0, 
                             callback=None)
params = bfgs_info[0]

KxX = get_K(x,X,params,num_IS)
KXX = get_K(X,X,params,num_IS)
mu1 = KxX @ np.linalg.pinv(KXX) @ Y
sigma2_1 = np.sqrt(get_sigma2(x,X,params,num_IS))

n = 100
x = np.zeros([n,p+1]) + 1
x[:,1] = np.linspace(0,1,n)

KxX = get_K(x,X,params,num_IS)
KXX = get_K(X,X,params,num_IS)
mu2 = KxX @ np.linalg.pinv(KXX) @ Y

n = 100
x = np.zeros([n,p+1]) + 2
x[:,1] = np.linspace(0,1,n)

KxX = get_K(x,X,params,num_IS)
KXX = get_K(X,X,params,num_IS)
mu3 = KxX @ np.linalg.pinv(KXX) @ Y

plt.figure(1)
plt.plot(X[X[:,0]==0,1],Y[X[:,0]==0],'bs')
plt.plot(X[X[:,0]==1,1],Y[X[:,0]==1],'rs')
plt.plot(X[X[:,0]==2,1],Y[X[:,0]==2],'gs')
plt.plot(x[:,1],mu1,'b')
plt.plot(x[:,1],mu2,'r')
plt.plot(x[:,1],mu3,'g')
plt.xlabel('x')
plt.ylabel('y')

# Real Process
n = 100
x = np.zeros([n,p+1])
x[:,1] = np.linspace(0,1,n)
y = get_y(x)
plt.plot(x[:,1],y,'b',linewidth = 3)

#Standard Deivation on Primary GP
plt.legend(['Data 0','Data 1','Data 2'])

#%% Regular GP Control

Y1 = Y[:N1]
X1 = X[:N1,:]

#Get Testing Data
n = 100
x = np.zeros([n,p+1])
x[:,1] = np.linspace(0,1,n)

#Define Hyperparameters and Bounds
param = np.array([1,0.5,0.5, #model 0: sigma, sigmaf, lambda
                    1,1,0.5,  #model 1: sigma, sigmaf, lambda
                    1,1,0.5]) 

bfgs_info = scipy.optimize.fmin_l_bfgs_b(func = get_like, 
                              x0 = param, args=(X1,Y1,num_IS), 
                              approx_grad=True,
                              bounds = param_bounds, 
                              m=10, 
                              factr=10.0, 
                              pgtol=0.01,
                              epsilon=1e-08, 
                              iprint=-1, 
                              maxfun=15000, 
                              maxiter=100, 
                              disp=0, 
                              callback=None)
params = bfgs_info[0]

KxX = get_K(x,X1,params,num_IS)
KXX = get_K(X1,X1,params,num_IS)
mu4 = KxX @ np.linalg.pinv(KXX) @ Y1
sigma2 = np.sqrt(get_sigma2(x,X1,params,num_IS))

plt.figure(2)
plt.plot(x[:,1],mu4,'g')
plt.plot(x[:,1],mu4+sigma2,'g--')
plt.plot(x[:,1],mu4-sigma2,'g--')
plt.plot(x[:,1],mu1,'b')
plt.plot(x[:,1],mu1+sigma2_1,'b--')
plt.plot(x[:,1],mu1-sigma2_1,'b--')
plt.plot(x[:,1],y,'k',linewidth = 3)
plt.xlabel('x')
plt.ylabel('y')

#%% Part 3 Example Problem using GA

X_list,Y_list = get_GA(get_like_priors,9,X,Y,num_IS)

#%%

#Plot Everything
iterations = np.arange(len(Y_list))
y_best = np.zeros(len(Y_list))
y_best_overall = np.zeros(len(Y_list))
for i in range(len(Y_list)):
    y_best[i] = min(Y_list[i])
    ind_y = np.argmin(Y_list[i])
    if i == 0:
        y_best_overall[i] = y_best[i]
        X_best = X_list[i][ind_y]
    else:
        if y_best[i] < min(y_best[:i]):
            y_best_overall[i] = y_best[i]
            X_best = X_list[i][ind_y]
        else:
            y_best_overall[i] = y_best_overall[i-1]

plt.figure()
plt.plot(iterations,y_best)
plt.plot(iterations,y_best_overall)
plt.xlabel('Iteration of GA')
plt.legend(['Best Point','Cumulative Best Point'])

params = X_best

KxX = get_K(x,X,params,num_IS)
KXX = get_K(X,X,params,num_IS)
mu1 = KxX @ np.linalg.pinv(KXX) @ Y

n = 100
x = np.zeros([n,p+1]) + 1
x[:,1] = np.linspace(0,1,n)

KxX = get_K(x,X,params,num_IS)
KXX = get_K(X,X,params,num_IS)
mu2 = KxX @ np.linalg.pinv(KXX) @ Y

n = 100
x = np.zeros([n,p+1]) + 2
x[:,1] = np.linspace(0,1,n)

KxX = get_K(x,X,params,num_IS)
KXX = get_K(X,X,params,num_IS)
mu3 = KxX @ np.linalg.pinv(KXX) @ Y

plt.figure()
plt.plot(X[X[:,0]==0,1],Y[X[:,0]==0],'bs')
plt.plot(X[X[:,0]==1,1],Y[X[:,0]==1],'rs')
plt.plot(X[X[:,0]==2,1],Y[X[:,0]==2],'gs')
plt.plot(x[:,1],mu1,'b')
plt.plot(x[:,1],mu2,'r')
plt.plot(x[:,1],mu3,'g')
plt.xlabel('x')
plt.ylabel('y')

# Real Process
n = 100
x = np.zeros([n,p+1])
x[:,1] = np.linspace(0,1,n)
y = get_y(x)
plt.plot(x[:,1],y,'b',linewidth = 3)

#Standard Deivation on Primary GP
plt.legend(['Data 0','Data 1','Data 2'])