# Zachary Cosenza
# Post 18 Multi-Fidelity Model Toolbox Development

# TODO ----------------------
# Priors (noted lambda 0 is too high --> flat)
# HMC Hyperparameter Training
# Robust Toy Problems
# Gradient of q-KG
# q-Expected Improvement
# Heteroskedacity?
# ---------------------------

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy.optimize import minimize
import seaborn as sea
import time
from tqdm import tqdm

import torch
import gpytorch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models import SingleTaskGP
from botorch.models.multitask import MultiTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from gpytorch.constraints import Positive

import copy
def get_f(x):
    #Initalize
    y = np.zeros(x.shape[0])
    #Get Fidelities
    ind_0 = x[:,-1] == 0
    ind_1 = x[:,-1] == 1
    ind_2 = x[:,-1] == 2
    #Underlying Function
    y[ind_0] = np.sum((x[ind_0,0:5] - 0.5)**2,axis = 1) + np.sum(x[ind_0,5:10],axis = 1)
    y[ind_0] = y[ind_0] + np.random.uniform(0,0.1,size = len(y[ind_0]))
    #Add Fidelity 1
    y[ind_1] = np.sum((x[ind_1,0:5] - 0.5)**2,axis = 1) + np.sum(x[ind_1,5:10],axis = 1)
    y[ind_1] = y[ind_1] + np.random.uniform(0,0.25,size = len(y[ind_1]))
    #Add Fidleity 2
    y[ind_2] = np.sum((x[ind_2,0:2] - 0.5)**2,axis = 1) + np.sum(x[ind_2,0:2],axis = 1)
    y[ind_2] = y[ind_2] + np.random.uniform(0,0.1,size = len(y[ind_2]))
    return -y

#Generate Training and Testing Data
N = 20
n = 400
n_reps = 1
px = 3 #not including fidelity
X = np.random.uniform(0,1,size = (N,px+1))
X = np.repeat(X,n_reps,axis = 0) #replicates
x = np.random.uniform(0,1,size = (n,px+1))
task = np.random.choice(3,X.shape[0],p=[0.2,0.4,0.4])
X[:,-1] = task
x[:,-1] = 0 #set testing data to fidelity 0
Y = get_f(X)
common_std = Y.std()
common_mean = Y.mean()
Y = (Y - common_mean) / common_std
y = get_f(x)
y = (y - common_mean) / common_std
num_IS = 3

def con(p, p_start = 0, p_end = 1):
    """
    Generate a constraint that has argument dependence
    p -- whatever p is
    p_start -- the first index for the constraint
    p_end -- the second index for the constraint
    """
    return p[int(p_start)] - p[int(p_end)]

def generate_constraints(x_size):
    """
    generate a list of constraints based on:
    n_con -- number of constraints
    con_args -- a n_con x 2 list of parameters to be passed to con
    """
    ind = np.arange(x_size)
    
    ind_0 = ind[0:px*q0]
    ind_1 = ind[px*q0:px*(q0+q0)]
    ind_2 = ind[px*(q+q0):px*(q+q0+q0)]
    
    ind_00 = ind[px*(q0+q0):px*(q+q0)]
    ind_3 = ind[px*(q+q0+q0):px*(q+q+q0)]
    
    n_con = len(ind_1) + len(ind_2) + len(ind_3)
    
    con_args = np.zeros([n_con,2])
    con_args[:,0] = np.hstack((ind_0,ind_0,ind_00)).flatten()
    con_args[:,1] = np.hstack((ind_1,ind_2,ind_3)).flatten()
    
    con_list = []
    con_dict = {'type':'eq'}
    for i in range(n_con):
        i_dict = copy.deepcopy(con_dict)
        i_dict['fun'] = con
        i_dict['args'] = [con_args[i][0], con_args[i][1]]
        con_list.append(i_dict)
        
    return con_list

def get_opt_qKG(q,q0,Nxx,X,Y,params):
    #Input is Unwrapped
    x_size = (q + q + q0 + Nxx) * px
    x_bounds = []
    for i in range(x_size):
        x_bounds.append((0,1))
    M = 5
    x = np.zeros([M,x_size])
    aq = np.zeros(M)
    con = generate_constraints(x_size)
    for i in tqdm(np.arange(M)):
        x0 = np.random.uniform(low = 0, high = 1, size = x_size)
        # bfgs = scipy.optimize.fmin_l_bfgs_b(func=wrapper_qKG, x0=x0,args=(q,q0,Nxx,X,Y,params), approx_grad=True,
        #                                 bounds=x_bounds, m=10, factr=10.0, pgtol=0.0001,
        #                                 epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=1000, 
        #                                 disp=0, callback=None)
        # x[i,:] = bfgs[0]
        # aq[i] = wrapper_qKG(x[i,:],q,q0,Nxx,X,Y,params)
        slsqp = minimize(fun = wrapper_qKG,
                        x0 = x0,
                        args = (q,q0,Nxx,X,Y,params),
                        method = 'SLSQP',
                        bounds = x_bounds,
                        constraints = con)
        x[i,:] = slsqp['x']
        aq[i] = slsqp['fun']
    ind_kg = np.argmin(aq)
    x_opt = x[ind_kg,:]
    x_opt = x_opt.reshape((q + q + q0 + Nxx),px)
    tasks = np.vstack((np.zeros([q0,1]),np.ones([q,1]),2*np.ones([q,1])))
    x_opt_query = np.hstack((x_opt[0:(q + q + q0),:],tasks))
    x_fantasy = np.hstack((x_opt[(q + q + q0):,:],np.zeros([Nxx,1])))
    return x_opt_query,x_fantasy

def wrapper_qKG(p,q,q0,Nxx,X,Y,params):
    #Re-Factor Vector into Matrix of Queries
    P = p.reshape((q + q + q0 + Nxx),px)
    #Generate Real Points with Constraint Fidelities
    tasks = np.vstack((np.zeros([q0,1]),np.ones([q,1]),2*np.ones([q,1])))
    x = np.hstack((P[0:(q + q + q0),:],tasks))
    #Generate Fantasy Points with Fidelity 0
    xx = P[(q + q + q0):,:]
    xx = np.hstack((xx,np.zeros([Nxx,1])))
    #Get Knowledge Gradient
    KG = get_qKG(x,xx,X,Y,params)
    return -KG

def get_chol_samples(x,mu,C,X,Y,M,params):
    L = np.linalg.cholesky(C + np.eye(len(mu)) * 0.00001)
    np.random.seed(1)
    z = np.random.normal(0,1,size = (len(mu),M))
    y_sample = mu.reshape(len(mu),1) + L @ z
    return y_sample

def get_cond_samples(x,mu_x,X,params):
    #x is point you want to predict
    #mu_x is prediction of that point
    #X is what is being conditioned on
    np.random.seed(1)
    KXX = get_K(X,X,params) + np.eye(X.shape[0]) * params[0]**2
    D = np.linalg.cholesky(KXX)
    D_inv = np.linalg.pinv(D)
    KxX = get_K(x,X,params)
    M = 1000
    Z = np.random.normal(0,1,size = (X.shape[0],M))
    mu = mu_x + KxX @ D_inv @ Z
    return np.mean(mu,axis = 1)

def get_qKG(x,xx,X,Y,params):
    # #Direct Method (For Testing)
    # mu_,_ = get_mu(x,X,Y,params)
    # X_ = np.vstack((X,x))
    # Y_ = np.hstack((Y,mu_))
    # mu,_ = get_mu(xx,X_,Y_,params)
    # q_KG = np.mean(mu)
    
    # #Semi-Sampling Method
    # #x Actual Point
    # #xx Fantasy Point at Fidelity = 0
    # mu_,_ = get_mu(x,X,Y,params)
    # C = get_C(x,X,Y,params)
    # #Get Nxx Samples of Posterior
    # M = xx.shape[0]
    # yy = get_chol_samples(x,mu_,C,X,Y,M,params)
    # #For Each Sample, Predict Each Fantasy
    # qkg_per_fant = np.zeros(M)
    # for i in np.arange(M):   
    #     X_ = np.vstack((X,x))
    #     Y_ = np.hstack((Y,yy[:,i]))
    #     x_fant = xx[i,:].reshape(1,xx.shape[1])
    #     mu,_ = get_mu(x_fant,X_,Y_,params)
    #     qkg_per_fant[i] = mu
    # #Average of Fantasy Points (Balandat et al)
    # q_KG = np.mean(qkg_per_fant)
    
    #Joint-Sampling Method
    M = 1
    mu_,_ = get_mu(xx,X,Y,params)
    C = get_C(xx,X,Y,params)
    mu_x = get_chol_samples(xx,mu_,C,X,Y,M,params)
    # mu_x,_ = get_mu(xx,X,Y,params)
    qkg_per_fant = np.zeros(xx.shape[0])
    for i in np.arange(xx.shape[0]):   
        x_fant = xx[i,:].reshape(1,xx.shape[1])
        qkg_per_fant[i] = get_cond_samples(x_fant,mu_x[i],x,params)
    q_KG = np.mean(qkg_per_fant)
    return q_KG

def get_Lgrad(params,X,Y):
    Ky = get_K(X,X,params) + np.eye(X.shape[0]) * params[0]**2
    Ky_inv = np.linalg.pinv(Ky)
    Lgrad = np.zeros(len(params))
    for i in np.arange(len(params)):
        if i == 0:
            #Noise
            grad = get_dKdsigma(X,params)
        elif i <= 3 and i > 0:
            #Output Lengthscale
            grad = get_dKdsigmaf(X,params,i)
        else:
            #Input Lengthscale
            grad = get_dKdlam(X,params,i)
        A = 0.5 * Y.T @ Ky_inv @ grad @ Ky_inv @ Y - 0.5 * np.trace(Ky_inv @ grad)
        Lgrad[i] = A
    #Use Normal Prior
    log_prior = get_prior_like_grad(params)
    Lgrad = Lgrad + log_prior
    return np.array(Lgrad)

def get_dKdsigma(X,params):
    dKdsigma = np.eye(X.shape[0]) * params[0] * 2
    return dKdsigma

def get_dKdsigmaf(X,params,k):
    num_IS = 3
    sigmaf = params[1:num_IS+1]
    lambdas = params[num_IS+1:]
    dKdsigmaf = np.zeros([X.shape[0],X.shape[0]])
    for i in np.arange(X.shape[0]):
        IS = int(X[i,-1])
        for j in np.arange(X.shape[0]):
            d = (X[i,:-1]-X[j,:-1]) @ (X[i,:-1]-X[j,:-1]).T
            #Primary Kernel
            if k == 1:
                L0 = lambdas[0]
                dKdsigmaf[i,j] = 2 * sigmaf[0] * np.exp(-d / L0**2 / 2)
            #Deviation Kernel
            if IS != 0 and IS == X[j,-1]:
                if k == 2:
                    L = lambdas[1]
                    dKdsigmaf[i,j] = 2 * sigmaf[1] * np.exp(-d / L**2 / 2)
                elif k == 3:
                    L = lambdas[2]
                    dKdsigmaf[i,j] = 2 * sigmaf[2] * np.exp(-d / L**2 / 2)  
    return dKdsigmaf

def get_dKdlam(X,params,k):
    num_IS = 3
    sigmaf = params[1:num_IS+1]
    lambdas = params[num_IS+1:]
    dKdsigmaf = np.zeros([X.shape[0],X.shape[0]])
    for i in np.arange(X.shape[0]):
        IS = int(X[i,-1])
        for j in np.arange(X.shape[0]):
            d = (X[i,:-1]-X[j,:-1]) @ (X[i,:-1]-X[j,:-1]).T
            #Primary Kernel
            if k == 4:
                L0 = lambdas[0]
                dKdsigmaf[i,j] = sigmaf[0]**2 * np.exp(-d / L0**2 / 2) * d / L0**3
            #Deviation Kernel
            if IS != 0 and IS == X[j,-1]:
                if k == 5:
                    L = lambdas[1]
                    dKdsigmaf[i,j] = sigmaf[1]**2 * np.exp(-d / L**2 / 2) * d / L**3
                elif k == 6:
                    L = lambdas[2]
                    dKdsigmaf[i,j] = sigmaf[2]**2 * np.exp(-d / L**2 / 2) * d / L**3  
    return dKdsigmaf

def get_dKdx(x,X,Y,params):
    num_IS = 3
    sigmaf = params[1:num_IS+1]
    lambdas = params[num_IS+1:]
    K = np.zeros([x.shape[0],X.shape[0]])
    for i in np.arange(x.shape[0]):
        IS = int(x[i,-1])
        for j in np.arange(X.shape[0]):
            d = (x[i,:-1]-X[j,:-1]) @ (x[i,:-1]-X[j,:-1]).T
            #Primary Kernel
            L0 = lambdas[0]
            K[i,j] = sigmaf[0]**2 * np.exp(-d / L0**2 / 2)
            #Deviation Kernel
            if IS != 0 and IS == X[j,-1]:
                L = lambdas[IS]
                K[i,j] = K[i,j] + sigmaf[IS]**2 * np.exp(-d / L**2 / 2)
    return K

def get_C(x,X,Y,params):
    Kxx = get_K(x,x,params)
    KxX = get_K(x,X,params)
    KXX = get_K(X,X,params)
    C = Kxx - KxX @ np.linalg.pinv(KXX + params[0]**2 * np.eye(X.shape[0])) @ KxX.T
    return C

def get_mu(x,X,Y,params):
    KXX = get_K(X,X,params)
    KxX = get_K(x,X,params)
    Kxx = get_K(x,x,params)
    Ky_inv = np.linalg.pinv(KXX + params[0]**2 * np.eye(X.shape[0]))
    mu = KxX @ Ky_inv @ Y
    cov = Kxx - KxX @ Ky_inv @ KxX.T
    sigma2 = np.diag(cov)
    return mu,sigma2

def get_K(x,X,params):
    num_IS = 3
    sigmaf = params[1:num_IS+1]
    lambdas = params[num_IS+1:]
    K = np.zeros([x.shape[0],X.shape[0]])
    for i in np.arange(x.shape[0]):
        IS = int(x[i,-1])
        for j in np.arange(X.shape[0]):
            d = (x[i,:-1]-X[j,:-1]) @ (x[i,:-1]-X[j,:-1]).T
            #Primary Kernel
            L0 = lambdas[0]
            K[i,j] = sigmaf[0]**2 * np.exp(-d / L0**2 / 2)
            #Deviation Kernel
            if IS != 0 and IS == X[j,-1]:
                L = lambdas[IS]
                K[i,j] = K[i,j] + sigmaf[IS]**2 * np.exp(-d / L**2 / 2)
    return K

def get_like(params,X,Y):
    KXX = get_K(X,X,params)
    Ky = KXX + params[0]**2 * np.eye(X.shape[0])
    logp = -0.5 * Y.T @ np.linalg.pinv(Ky) @ Y - 0.5 * np.log(np.linalg.det(Ky)) 
    - n/2*np.log(2*np.pi) + get_prior_like(params)
    return logp

def get_prior_like(p):
    #Get Prior for Lambda Parameter
    means = np.array([1,1,1])
    sigs = np.diag((means/2)**2)
    #Return Normal Prior of Log Likelihood
    S = np.linalg.inv(sigs)
    x_mu = p[num_IS+1:] - means
    log_prior = -0.5 * x_mu @ S @ x_mu
    return log_prior

def get_prior_like_grad(p):
    #Get Prior for Lambda Parameter
    log_prior_grad = np.zeros(len(p))
    means = np.array([1,1,1])
    sigs = np.diag((means/2)**2)
    #Return Normal Prior of Log Likelihood
    S = np.linalg.inv(sigs)
    x_mu = p[num_IS+1:] - means
    #Get Gradient
    log_prior_lambda_grad = -0.5 * (S + S.T) @ x_mu
    log_prior_grad[num_IS+1:] = log_prior_lambda_grad
    return log_prior_grad

def wrapper_like(p,X,Y):
    return -get_like(p,X,Y)

def wrapper_glike(p,X,Y):
    return -get_Lgrad(p,X,Y)

def get_opt_params(X,Y):
    num_hypers = 2 * num_IS + 1
    param_low = 0.01
    param_high = 10
    param_bounds = []
    for i in range(num_hypers):
        bound = (param_low,param_high)
        param_bounds.append(bound)
    M = 35
    params = np.zeros([M,num_hypers])
    like = np.zeros(M)
    for i in tqdm(np.arange(M)):
        params0 = np.random.uniform(low = param_low, high = 1, size = num_hypers)
        bfgs = scipy.optimize.fmin_l_bfgs_b(func=wrapper_like, x0=params0, fprime = wrapper_glike, 
                                            args=(X,Y), approx_grad=False,
                                        bounds=param_bounds, m=10, factr=10.0, pgtol=0.0001,
                                        epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=1000, 
                                        disp=0, callback=None)
        params[i,:] = bfgs[0]
        like[i] = get_like(params[i,:],X,Y)
    ind_like = np.argmax(like)
    params_opt = params[ind_like,:]
    return params_opt

# Solution to GP Problem
# params = get_opt_params(X,Y)
params = np.array([0.0360055,0.3983522,0.0582616,2.19184049,0.95576314,1.42709205,4.16834235])
y_custom,std2_custom = get_mu(x,X,Y,params)

# Test of q-KG
q = 2
q0 = 1
Nxx = 5

# Test of q-KG Optimization
start = time.time()
x_opt_query,x_fantasy = get_opt_qKG(q,q0,Nxx,X,Y,params)
end = time.time()
print(end - start)
print(x_opt_query)
print(x_fantasy)

def train(X,Y,model,likelihood,training_iter=100):
    X = torch.tensor(X,dtype=torch.double)
    Y = torch.tensor(Y,dtype=torch.double)
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameter
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(X)
        loss = -mll(output,Y)
        loss.backward()
        optimizer.step()
    model.eval()
    likelihood.eval()

def predict(model,likelihood,x):
    x = torch.tensor(x)
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        return likelihood(model(x))

class CustomKernel(gpytorch.kernels.Kernel):
    def __init__(self):
        super().__init__()
        self.register_parameter(name='raw_length', 
                                parameter=torch.nn.Parameter(torch.zeros(num_IS, 1)))
        self.register_parameter(name='raw_outputscale',
                                parameter=torch.nn.Parameter(torch.zeros(num_IS, 1)))
        length_constraint = Positive()
        outputscale_constraint = Positive()
        self.register_constraint("raw_length", length_constraint)
        self.register_constraint('raw_outputscale', outputscale_constraint)

    @property
    def length(self):
        return self.raw_length_constraint.transform(self.raw_length)
    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)    

    @length.setter
    def length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        return self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))
    @outputscale.setter
    def outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        return self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))   
    
    # this is the kernel function
    def forward(self, x1, x2, **params):
        diff = torch.tensor(get_custom_covar_IS(self,x1,x2))
        diff.where(diff == 0, torch.as_tensor(1e-20).double())
        #potential problems:
            #need to standardize
            #something with the mean function
            #ARD kernel
        return diff

def get_custom_covar_IS(self,x1,x2):
    x1 = x1.numpy()
    x2 = x2.numpy()
    n,p = x1.shape
    N,_ = x2.shape
    K = np.zeros([n,N])
    for i in np.arange(n):
        IS = int(x1[i,-1])
        for j in np.arange(N):
            L0 = self.length.detach().numpy()[0]
            d = (x1[i,:-1]-x2[j,:-1])**2 / L0**2
            r = sum(d)
            K[i,j] = self.outputscale.detach().numpy()[0] * np.exp(-r)
            if IS != 0 and IS == x2[j,-1]:
                L = self.length.detach().numpy()[IS]
                d = (x1[i,:-1]-x2[j,:-1])**2 / L**2
                r = sum(d)
                K[i,j] = K[i,j] + self.outputscale.detach().numpy()[IS] * np.exp(-r)
    return K

# Use the simplest form of GP model, exact inference
class CustomModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = CustomKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# like_custom = gpytorch.likelihoods.GaussianLikelihood()
# model_custom = CustomModel(torch.tensor(X),
#                             torch.tensor(Y),like_custom)
# train(X,Y,model_custom,like_custom)
# y_bo_custom = predict(model_custom,like_custom,x).mean

# model_fid = SingleTaskMultiFidelityGP(
#     torch.tensor(X), 
#     torch.tensor(Y).unsqueeze(-1),
#     data_fidelity=px)
# model = SingleTaskGP(torch.tensor(X[X[:,-1]==0,:-1]),
#                       torch.tensor(Y[X[:,-1]==0]).unsqueeze(-1))
# mll_fid = ExactMarginalLogLikelihood(model_fid.likelihood, model_fid)
# mll_fid = fit_gpytorch_model(mll_fid)
# mll = ExactMarginalLogLikelihood(model.likelihood,model)
# mll = fit_gpytorch_model(mll)
# model_task = MultiTaskGP(torch.tensor(X),torch.tensor(Y).unsqueeze(-1),task_feature=-1)

# with torch.no_grad():
    # y_bo_fid = model_fid.posterior(torch.tensor(x)).mean.cpu().numpy()
    # y_bo = model.posterior(torch.tensor(x[x[:,-1]==0,:-1])).mean.cpu().numpy()
    # y_task = model_task.posterior(torch.tensor(x[:,:-1])).mean.cpu().numpy()

# X_linear = X[X[:,-1]==0,:-1]
# x_linear = x[:,:-1]
# beta_linear = X_linear.T @ np.linalg.pinv(X_linear @ X_linear.T) @ Y[X[:,-1]==0]
# y_linear = beta_linear @ x_linear.T
# X_poly = np.hstack((np.ones([X_linear.shape[0],1]),X_linear,X_linear**2))
# x_poly = np.hstack((np.ones([x_linear.shape[0],1]),x_linear,x_linear**2))
# beta_poly = X_poly.T @ np.linalg.pinv(X_poly @ X_poly.T) @ Y[X[:,-1]==0]
# y_poly = beta_poly @ x_poly.T

plt.figure()
plt.subplot(1,2,1)
markersize = 2
plt.plot(x[:,0],y,'k.',markersize=markersize)
plt.plot(X[X[:,-1]==0,0],Y[X[:,-1]==0],'ks')
plt.plot(X[X[:,-1]==1,0],Y[X[:,-1]==1],'k+')
plt.plot(X[X[:,-1]==2,0],Y[X[:,-1]==2],'k*')
# plt.plot(x[:,0],y_bo_fid,'b.',markersize=markersize)
# plt.plot(x[x[:,-1]==0,0],y_bo,'b.',markersize=markersize)
# plt.plot(x[:,0],y_task[:,0],'y.',markersize=markersize)
# plt.plot(x[:,0],y_linear,'c.',markersize=markersize)
# plt.plot(x[:,0],y_poly,'g.',markersize=markersize)
plt.plot(x[:,0],y_custom,'r.',markersize=markersize)
# plt.plot(x[:,0],y_bo_custom,'b.',markersize=markersize)
# plt.xlabel('x')
# plt.ylabel('y')

plt.subplot(1,2,2)
# plt.plot(y,y_bo_fid,'b.',markersize=markersize)
# plt.plot(y,y_bo,'b.',markersize=markersize)
# plt.plot(y,y_task[:,0],'y.',markersize=markersize)
# plt.plot(y,y_linear,'c.',markersize=markersize)
# plt.plot(y,y_poly,'g.',markersize=markersize)
# plt.plot(y,y_bo_custom,'b.',markersize=markersize)
plt.plot(y,y_custom,'r.',markersize=markersize)
plt.plot(y,y,'k',markersize=markersize)
# plt.xlabel('y true')
# plt.ylabel('y pred')

# print("MSE Botorch MultiFidelity: {}".format(sum((y_bo_fid-y.reshape(-1,1))**2)))
# print("MSE Botorch GP: {}".format(sum((y_bo-y.reshape(-1,1))**2)))
# print("MSE Custom Botorch GP: {}".format(sum((y_bo_custom-y)**2)))
# print("MSE Botorch MultiTask: {}".format(sum((y_task[:,0]-y)**2)))
# print("MSE Linear OLS: {}".format(sum((y_linear-y)**2)))
# print("MSE Polynomial OLS: {}".format(sum((y_poly-y)**2)))
# print("MSE Custom Multi-Fidelity GP: {}".format(sum((y_custom-y)**2)))