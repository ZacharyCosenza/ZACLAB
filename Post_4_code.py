# Zachary Cosenza
# Post 4 Code

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

#Example of Bayes Rule

X = np.linspace(0,1,5)
x = np.linspace(0,1,100)
Y = 0.5 * X
y = 0.5 * x
e = np.random.normal(0,Y/2,size = len(X))
Y = Y + e

plt.figure()
plt.plot(X,Y,'rs')
plt.plot(x,y,'k')

#Define Prior
slope_mean = 0
slope_std = 1
slopes = np.linspace(-2,2,100)
prior = norm.pdf(slopes, loc = slope_mean, scale = slope_std)

#Define Likelihood Function
error_mean = 0
error_std = 1
like = [np.prod(norm.pdf(slope * X - Y, loc = error_mean, scale = error_std)) for slope in slopes]

#Define Posterior
post = like * prior

plt.figure()
plt.plot(slopes,like)

plt.figure()
plt.plot(slopes,prior)

plt.figure()
plt.plot(slopes,post)

#%% Part 1 Gibbs Sampling

# 2 Uncorrelated Normal Distributions
theta = np.array([0.5,0.5])
mu_0 = 1
mu_1 = 2
sigma_0 = 1
sigma_1 = 0.5

p = len(theta)
K = 1000 # number of samples
theta_gibbs = np.zeros([K*p,p])

j = 0
for k in np.arange(K):
    for i in np.arange(p):
        if i == 0:
            mu = mu_0
            sigma = sigma_0
        else:
            mu = mu_1
            sigma = sigma_1
        theta[i] = np.random.normal(mu,sigma)
        theta_gibbs[j,:] = theta
        j = j + 1

plt.figure()
plt.plot(theta_gibbs[:,0],theta_gibbs[:,1],'k.')
plt.plot(mu_0,mu_1,'rs')
plt.xlabel('theta0')
plt.ylabel('theta1')

plt.figure()
plt.subplot(1,2,1)
plt.hist(theta_gibbs[:,0])
plt.title('theta0 Distribution')

plt.subplot(1,2,2)
plt.hist(theta_gibbs[:,1])
plt.title('theta1 Distribution')

# 2 Uncorrelated Binomial and Beta Distributions
theta = np.array([0.5,0.5])
n = 5
alpha = 2
beta = 2

p = len(theta)
theta_gibbs = np.zeros([K*p,p])
theta_gibbs[0,:] = theta

j = 0
for k in np.arange(K-1):
    for i in np.arange(p):
        if i == 0:
            y = theta_gibbs[k,1]
            theta[i] = np.random.binomial(n,y)
        else:
            x = theta_gibbs[k,0]
            theta[i] = np.random.beta(x+alpha,n-x+beta)
        theta_gibbs[j+1,:] = theta
        j = j + 1
        
plt.figure()
plt.plot(theta_gibbs[:,0],theta_gibbs[:,1],'k.')
plt.xlabel('y')
plt.ylabel('x')

plt.figure()
plt.subplot(1,2,1)
plt.hist(theta_gibbs[:,0],bins = 6)
plt.title('y Distribution')

plt.subplot(1,2,2)
plt.hist(theta_gibbs[:,1])
plt.title('x Distribution')

# 4 Correlated Distributions based on Data
# (Not a fan of how some parameters are modeled)
# import pandas as pd
# from scipy.stats import chi2

# dfA = pd.DataFrame({'A': [62,60,63,50]})
# dfB = pd.DataFrame({'B':[63,67,71,64,65,66]})
# dfC = pd.DataFrame({'C':[68,66,71,67,68,68]})
# dfD = pd.DataFrame({'D':[56,62,60,61,63,64,63,59]})

# Y = pd.concat([dfA,dfB,dfC,dfD], ignore_index=True, axis=1)

# J = len(Y.columns)
# nj = Y.count()
# n = sum(nj)
# K = 1000
# p = 7

# #Initalize Parameters
# theta_gibbs = np.zeros([K*p,p])
# theta = np.random.uniform(low = 0, high = 100, size = 7)
# theta_gibbs[0,:] = theta

# j = 0
# for k in np.arange(K-1):
#     for i in np.arange(p):
#         #Condition over theta_j 0,1,2,3 (theta[0:3])
#         if i <= 3:
#             t2 = theta[6]
#             sigma2 = theta[5]
#             y_hat = np.mean(Y.loc[:,i])
#             theta_hat_j = (1/t2*mu+nj[i]/sigma2*y_hat)/(1/t2+nj[i]/sigma2)
#             V_hat = 1/(1/t2+nj[i]/sigma2)
#             theta[i] = np.random.normal(theta_hat_j,V_hat)
#         #Condition over mu (theta[4])
#         elif i == 4:
#             mu_hat = np.mean(theta[0:3])
#             t2 = theta[6]
#             theta[i] = np.random.normal(mu_hat,t2/J)
#         #Conditon over sigma**2 (theta[5])
#         elif i == 5:
#             A = np.zeros(Y.shape)
#             for jj in np.arange(J): #select column of Y
#                 for ii in np.arange(nj[jj]): #select row of Y
#                     A[ii,jj] = (Y.loc[ii,jj] - theta[jj])**2
#             sigma2_hat = 1/n*np.sum(A)
#             invX2 = chi2.rvs(n)
#             theta[i] = invX2
#         #Condition over t**2 (theta[6])
#         elif i == 6:
#             mu = theta[4]
#             t2_hat = 1/(J-1)*np.sum((theta[0:3]-mu)**2)
#             invX2 = chi2.rvs(J-1)
#             theta[i] = invX2
#         theta_gibbs[j+1,:] = theta
#         j = j + 1
            
# plt.figure()
# for i in np.arange(p):
#     plt.subplot(2,4,i+1)
#     plt.hist(theta_gibbs[:,i])

# plt.figure()
# shave = np.arange(100,K)
# for i in np.arange(p):
#     plt.subplot(2,4,i+1)
#     plt.plot(np.arange(len(theta_gibbs[shave,i])),theta_gibbs[shave,i],linewidth=0.5)
    
#%% Part 2 Metropolis Hastings Algorithm
from scipy.stats import multivariate_normal

# 2 Uncorrelated Normal Distributions
theta = np.array([0.5,0.5])
mu_0 = 1
mu_1 = 2
MU = np.array([mu_0,mu_1])
SIGMA = np.array([[1,0.2],[0.2,1]])

p = len(theta)
K = 1000 # number of samples
theta_mh = np.zeros([K,p])

C = np.eye(p) * 0.2
posterior = multivariate_normal(mean=MU,cov=SIGMA)

for k in np.arange(K-1):
    #Generate Proposal
    theta_proposal = np.random.multivariate_normal(theta,C)
    proposalx = multivariate_normal(mean=theta,cov=C)
    proposaly = multivariate_normal(mean=theta_proposal,cov=C)
    
    px = posterior.pdf(theta)
    py = posterior.pdf(theta_proposal)
    Qxy = proposaly.pdf(theta)
    Qyx = proposalx.pdf(theta_proposal)
    
    #Calculate Alpha
    alpha = min([py/px*Qxy/Qyx,1])
    #Solve for theta_mh[k+1]
    if alpha > np.random.uniform(low = 0, high = 1):
        theta_mh[k+1] = theta_proposal
    else:
        theta_mh[k+1] = theta
    
    #Update theta
    theta = theta_mh[k+1]

plt.figure()
plt.plot(theta_mh[:,0],theta_mh[:,1],'k.')
plt.plot(mu_0,mu_1,'rs')
plt.xlabel('theta0')
plt.ylabel('theta1')

plt.figure()
plt.subplot(1,2,1)
plt.hist(theta_mh[:,0])
plt.title('theta0 Distribution')

plt.subplot(1,2,2)
plt.hist(theta_mh[:,1])
plt.title('theta1 Distribution')

plt.figure()
shave = np.arange(100,K)
for i in np.arange(p):
    plt.subplot(2,4,i+1)
    plt.plot(np.arange(len(theta_mh[shave,i])),theta_mh[shave,i],linewidth=0.5)