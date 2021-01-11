# Zachary Cosenza
# Post 6 Code

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def get_donner_post(beta,X,Y):
    prior = 1
    like = np.prod(norm.cdf(beta.T @ X) ** Y * (1 - norm.cdf(beta.T @ X)) ** (1 - Y))
    post = prior * like
    return post

# Make Fake Data
X = np.array([[1,1,0,0,0,0,1],[16,40,45,60,65,4,11]])
Y = np.array([1,0,0,1,0,1,1])
X = np.vstack((np.ones([1,7]),X))

# Test Probit Model
beta = np.array([0.5,1,-0.1])
ages = np.linspace(1,70,100).reshape(1,100)
x = np.vstack((np.ones([1,100]),np.ones([1,100]),ages))
z = beta.T @ x
y_test = norm.cdf(z).reshape(1,100)

plt.plot(ages,y_test,'r.')
plt.plot(X[2,:],Y,'rs')

#HMC Integration of Beta

p = len(beta)
K = 100 # number of samples
beta_sample = np.zeros([K,p])
C = np.eye(p)

from scipy.stats import multivariate_normal

for k in np.arange(K-1):
    #Generate Proposal
    beta_proposal = np.random.multivariate_normal(beta,C)
    proposalx = multivariate_normal(mean=beta,cov=C)
    proposaly = multivariate_normal(mean=beta_proposal,cov=C)
    
    px = get_donner_post(beta,X,Y)
    py = get_donner_post(beta_proposal,X,Y)
    
    Qxy = proposaly.pdf(beta)
    Qyx = proposalx.pdf(beta_proposal)
        
    #Calculate Alpha
    alpha = min([py/px*Qxy/Qyx,1])
    #Solve for theta_mh[k+1]
    if alpha > np.random.uniform(low = 0, high = 1):
        beta_sample[k+1] = beta_proposal
    else:
        beta_sample[k+1] = beta
    
    #Update beta
    beta = beta_sample[k+1]

plt.figure()
shave = np.arange(100,K)
for i in np.arange(p):
    plt.subplot(2,4,i+1)
    plt.plot(np.arange(len(beta_sample[shave,i])),beta_sample[shave,i],linewidth=0.5)

plt.figure()
plt.subplot(1,3,1)
plt.hist(beta_sample[:,0])
plt.subplot(1,3,2)
plt.hist(beta_sample[:,1])
plt.subplot(1,3,3)
plt.hist(beta_sample[:,2])

ages = np.linspace(1,70,100).reshape(1,100)
x = np.vstack((np.ones([1,100]),np.ones([1,100]),ages))

rand_seq = []
for i in range(K):
   r=np.random.randint(1,K)
   if r not in rand_seq: rand_seq.append(r)
          
plt.figure()
for k in rand_seq[0:int(K/100)]:
    z = beta_sample[k,:].T @ x
    y_test = norm.cdf(z).reshape(1,100)
    plt.plot(ages,y_test,'k.')
plt.plot(X[2,:],Y,'rs')
beta_mean = np.mean(beta_sample,axis = 0)
z_mean = beta_mean.T @ x
y_mean = norm.cdf(z_mean).reshape(1,100)
plt.plot(ages,y_mean,'r.')
plt.ylabel('Prob of Life')
plt.xlabel('Age')

#%% Part 2 Gaussian Process Model

from scipy.stats import multivariate_normal

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

def get_gpr_post(params,X,Y):
    n,p = X.shape
    sigma = params[0]
    Ky = get_K(X,X,params) + sigma**2 * np.eye(n)
    like = -0.5 * Y.T @ np.linalg.pinv(Ky) @ Y - 0.5 * np.log(np.linalg.det(Ky)) - n/2*np.log(2*np.pi)
    post = like
    if post <= 0:
        post = 10**-1
    return post

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

#HMC Integration of Beta

beta = params
p = len(beta)
K = 100 # number of samples
beta_sample = np.zeros([K,p])
C = np.eye(p) * 0.001

from scipy.stats import multivariate_normal

for k in np.arange(K-1):
    #Generate Proposal
    beta_proposal = np.random.multivariate_normal(beta,C)
    for i in range(p):
        if beta_proposal[i] <= 0:
            beta_proposal[i] = 0.01
    for i in range(p):
        if beta[i] <= 0:
            beta[i] = 0.01
        
    proposalx = multivariate_normal(mean=beta,cov=C)
    proposaly = multivariate_normal(mean=beta_proposal,cov=C)
    
    px = get_gpr_post(beta,X,Y)
    py = get_gpr_post(beta_proposal,X,Y)
    
    Qxy = proposaly.pdf(beta)
    Qyx = proposalx.pdf(beta_proposal)
        
    #Calculate Alpha
    alpha = min([py/px*Qxy/Qyx,1])
    #Solve for theta_mh[k+1]
    if alpha > np.random.uniform(low = 0, high = 1):
        beta_sample[k+1] = beta_proposal
    else:
        beta_sample[k+1] = beta
    
    #Update beta
    beta = beta_sample[k+1]

plt.figure()
plt.subplot(1,3,1)
plt.hist(beta_sample[:,0],int(K/4))
plt.subplot(1,3,2)
plt.hist(beta_sample[:,1],int(K/4))
plt.subplot(1,3,3)
plt.hist(beta_sample[:,2],int(K/4))

plt.figure()
shave = np.arange(100,K)
for i in np.arange(p):
    plt.subplot(2,4,i+1)
    plt.plot(np.arange(len(beta_sample[shave,i])),beta_sample[shave,i],linewidth=0.5)

beta_hmc = np.mean(beta_sample, axis = 0)

#Plot Contours
MakePlot(X,Y,beta_hmc,'mean')

#%% Part 3 Convergence Analysis

params = np.array([1,1,1])

M = 5 #Number of Chains
K = 100 #Number of Iterations

p = len(params)
beta_sample = np.zeros([K,p,M])
C = np.eye(p) * 0.001

for m in np.arange(M):
    beta = params
    for k in np.arange(K-1):
        #Generate Proposal
        beta_proposal = np.random.multivariate_normal(beta,C)
        for i in range(p):
            if beta_proposal[i] <= 0:
                beta_proposal[i] = 0.01
        for i in range(p):
            if beta[i] <= 0:
                beta[i] = 0.01
            
        proposalx = multivariate_normal(mean=beta,cov=C)
        proposaly = multivariate_normal(mean=beta_proposal,cov=C)
        
        px = get_gpr_post(beta,X,Y)
        py = get_gpr_post(beta_proposal,X,Y)
        
        Qxy = proposaly.pdf(beta)
        Qyx = proposalx.pdf(beta_proposal)
            
        #Calculate Alpha
        alpha = min([py/px*Qxy/Qyx,1])
        #Solve for theta_mh[k+1]
        if alpha > np.random.uniform(low = 0, high = 1):
            beta_sample[k+1,:,m] = beta_proposal
        else:
            beta_sample[k+1,:,m] = beta
        
        #Update beta
        beta = beta_sample[k+1,:,m]

beta_sample_1 = beta_sample[:round(K/2),:,:]
beta_sample_2 = beta_sample[round(K/2):,:,:]

beta_sample = np.concatenate((beta_sample_1,beta_sample_2),axis = 2)

plt.figure()
for i in np.arange(p):
    for j in np.arange(2*M):
        plt.subplot(2,4,i+1)
        plt.plot(np.arange(len(beta_sample[5:,i,j])),beta_sample[5:,i,j],linewidth=0.5)

beta_bar_j = np.zeros([p,M])
for j in np.arange(M):
    beta_bar_j[:,j] = np.mean(beta_sample[:,:,j],axis=0)

beta_bar = np.zeros(p)
for i in np.arange(p):
    beta_bar[i] = np.mean(beta_bar_j[i,:],axis=0)
    
s2 = np.zeros([p,M])
for j in np.arange(M):
    s2[:,j] = 1/(K-1)*np.sum((beta_sample[:,:,j]-beta_bar_j[:,j])**2,axis=0)

B = K/(M-1)*np.sum((beta_bar_j.T - beta_bar)**2,axis=0)
W = 1/M*np.sum(s2,axis=1)

Var = (K-1)/K*W + 1/K*B

R_hat = np.sqrt(Var/W)













