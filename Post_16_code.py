# Zachary Cosenza
# Post 16 Monte Carlo for q Problems

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sea
import scipy.optimize

def get_y(x):
    y = (x[:,0] - 0.5)**2 + 0.25*x[:,0] - 1
    return y

def get_y_fid(x):
    #First Dimension = Linear
    #Second Dimension = Nothing
    #Third Dimension = Fidelity
    y = ((x[:,0] - 0.5)**2 + 0.25*x[:,0] - 1) + 1 * (x[:,-1] == 1) + 2 * (x[:,-1] == 2)
    return y.reshape(-1,1)

def get_mu(x,X,Y,params):
    N = X.shape[0]
    n = x.shape[0]
    y = np.zeros(n)
    sigma2 = np.zeros(n)
    Ky_inv = np.linalg.pinv(get_K(X,X,params) + np.eye(N) * params[1]**2)
    for i in range(n):
        xin = x[i,:].reshape(1,x.shape[1])
        KxX = get_K(xin,X,params)
        Kxx = get_K(xin,xin,params)
        y[i] = KxX @ Ky_inv @ Y
        sigma2[i] = Kxx - KxX @ Ky_inv @ KxX.T
    return y,sigma2

def get_K(x,X,params):
    N = X.shape[0]
    n = x.shape[0]
    K = np.zeros([n,N])
    for i in np.arange(n):
        for j in np.arange(N):
            lambdas = params[2]
            r = sum((x[i,:]-X[j,:])**2) / lambdas**2
            K[i,j] = params[0]**2 * np.exp(-r/2)
    return K

def get_C(x,X,Y,params):
    Kxx = get_K(x,x,params)
    KxX = get_K(x,X,params)
    KXX = get_K(X,X,params)
    C = Kxx - KxX @ np.linalg.pinv(KXX + params[0]**2 * np.eye(X.shape[0])) @ KxX.T
    return C

def get_2EI_map(x,X,Y,params):
    x1 = x
    x2 = x
    mu_star = max(Y)
    two_EI = np.zeros([x1.shape[0],x2.shape[0]])
    for i in np.arange(x1.shape[0]):
        for j in np.arange(x2.shape[0]):
            Xin = np.vstack((x1[i,:],x2[j,:]))
            Yin,_ = get_mu(Xin,X,Y,params)
            two_EI[i,j] = (Yin[0] - mu_star)*(mu_star <= Yin[0])*(Yin[1] <= Yin[0]) 
            + (Yin[1] - mu_star)*(mu_star <= Yin[1])*(Yin[1] >= Yin[0])
    return two_EI

def get_qEI_map(x,Y,params):
    x1 = x
    x2 = x
    qEI = np.zeros([x1.shape[0],x2.shape[0]])
    for i in np.arange(x1.shape[0]):
        for j in np.arange(x2.shape[0]):
            Xin = np.vstack((x1[i,:],x2[j,:]))
            qEI[i,j] = get_qEI(Xin,X,Y,params)
    return qEI

def get_qEI(x,X,Y,params):
    #Assume: always evaluate at x + 0.1
    x_additional = x + 0.1
    x = np.vstack((x,x_additional))
    M = 200
    C = get_C(x,X,Y,params)
    mu,_ = get_mu(x,X,Y,params)
    L = np.linalg.cholesky(C + np.eye(C.shape[0]) * .00001)
    qEI = np.zeros(M)
    for m in np.arange(M):
        Z = np.random.normal(0,1,size = L.shape[0])
        y = mu + L @ Z
        qEI[m] = max(y) - max(Y)
        if qEI[m] < 0:
            qEI[m] = 0
    qEI = np.mean(qEI)
    return qEI

def get_KG(x,X,Y,params):
    M = 200
    C = get_C(x,X,Y,params)
    mu,_ = get_mu(x,X,Y,params)
    L = np.linalg.cholesky(C + np.eye(C.shape[0]) * .00001)
    KG = np.zeros(M)
    for m in np.arange(M):
        Z = np.random.normal(0,1,size = L.shape[0])
        yy = mu + L @ Z
        KG[m] = max(yy)
        if KG[m] < 0:
            KG[m] = 0
    KG = np.mean(KG)
    return KG

def get_qKG(x,xx,X,Y,params):
    #x is actual point
    #xx is fantasy point
    mu,_ = get_mu(xx,X,Y,params)
    q_KG_per_fantasy = np.zeros(xx.shape[0])
    for i in np.arange(xx.shape[0]):
        X_ = np.vstack((X,xx[i,:]))
        Y_ = np.vstack((Y.reshape(len(Y),1)
                        ,mu[i]))
        q_KG_per_fantasy[i] = get_KG(x,X_,Y_,params)
    q_KG = np.mean(q_KG_per_fantasy)
    return q_KG

def get_qKG_map(x,xx,X,Y,params):
    x1 = x
    x2 = x
    qKG = np.zeros([x1.shape[0],x2.shape[0]])
    for i in np.arange(x1.shape[0]):
        for j in np.arange(x2.shape[0]):
            Xin = np.vstack((x1[i,:],x2[j,:]))
            qKG[i,j] = get_qKG(Xin,xx,X,Y,params)
    return qKG

def wrapper_qEI(p,X,Y,params):
    qEI = -get_qEI(p,X,Y,params)
    return qEI

def opt_qEI(X,Y,params):
    x_bounds = ((0,1),(0,1),(0,1),(0,1))
    B = 10
    alpha_bfgs = np.zeros(B)
    x_keep = np.zeros([B,len(x_bounds)])
    for b in np.arange(B):
        x0 = np.random.uniform(0,1,size = len(x_bounds))
        bfgs = scipy.optimize.fmin_l_bfgs_b(func=wrapper_qEI, x0=x0, args=(X,Y,params), approx_grad=True,
                                                bounds=x_bounds, m=10, factr=10.0, pgtol=0.0001,
                                                epsilon=1e-08, iprint=-1, maxfun=15000, maxiter=1000, 
                                                disp=0, callback=None)
        x_keep[b,:] = bfgs[0]
        alpha_bfgs[b] = get_qEI(bfgs[0],X,Y,params)
    
    ind_bfgs = np.argsort(alpha_bfgs)
    x_opt = x_keep[ind_bfgs[-1],:]
    print(x_opt)

#Training and Testing Data
N = 5
X = np.random.uniform(0,1,(N,2))
Y = get_y(X)
n = 30
x = np.zeros((n,2))
x[:,0] = np.linspace(0,1,n)
y = get_y(x)
params = np.array([6,1,1]) #pick some parameters for the gp
y_pred,_ = get_mu(x,X,Y,params)

#Plot Data
plt.figure()
plt.plot(X[:,0],Y,'rs')
plt.plot(x[:,0],y,'k.')
plt.plot(x[:,0],y_pred,'r.')
plt.xlabel('x')
plt.ylabel('y')

#Analytical 2-EI Function
two_EI = get_2EI_map(x,X,Y,params)
plt.figure()
sea.heatmap(two_EI)
plt.xlabel('x2')
plt.ylabel('x1')
plt.title('q = 2, EI using Analytical')

#q-EI Function with Optimization
q_EI = get_qEI_map(x,Y,params)
plt.figure()
sea.heatmap(q_EI)
plt.xlabel('x2')
plt.ylabel('x1')
plt.title('q = 2, EI using MC')

#q-KG Function
N_fant = 5
xx = np.random.uniform(0,1,(N_fant,2))
q_KG = get_qKG_map(x,xx,X,Y,params)
plt.figure()
sea.heatmap(q_KG)
plt.xlabel('x2')
plt.ylabel('x1')
plt.title('q = 2, KG using MC')

#%% Helper Function for Botorch

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from botorch.acquisition.acquisition import (
    AcquisitionFunction,
    OneShotAcquisitionFunction,
)
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient
from botorch.generation.gen import gen_candidates_scipy
from botorch.logging import logger
from botorch.optim.initializers import (
    gen_batch_initial_conditions,
    gen_one_shot_kg_initial_conditions,
)
from botorch.optim.stopping import ExpMAStoppingCriterion
from torch import Tensor

def optimize_acqf_mixed(
    acq_function: AcquisitionFunction,
    bounds: Tensor,
    q: int,
    num_restarts: int,
    raw_samples: int,
    fixed_features_list: List[Dict[int, float]],
    options: Optional[Dict[str, Union[bool, float, int, str]]] = None,
    inequality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    equality_constraints: Optional[List[Tuple[Tensor, Tensor, float]]] = None,
    post_processing_func: Optional[Callable[[Tensor], Tensor]] = None,
    batch_initial_conditions: Optional[Tensor] = None,
    sequential: bool = False,
) -> Tuple[Tensor, Tensor]:
    r"""Optimize over a list of fixed_features and returns the best solution.
    This is useful for optimizing over mixed continuous and discrete domains.
    Args:
        acq_function: An AcquisitionFunction
        bounds: A `2 x d` tensor of lower and upper bounds for each column of `X`.
        q: The number of candidates.
        num_restarts:  Number of starting points for multistart acquisition
            function optimization.
        raw_samples: Number of samples for initialization.
        fixed_features_list: A list of maps `{feature_index: value}`. The i-th
            item represents the fixed_feature for the i-th optimization.
        options: Options for candidate generation.
        inequality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) >= rhs`
        equality constraints: A list of tuples (indices, coefficients, rhs),
            with each tuple encoding an inequality constraint of the form
            `\sum_i (X[indices[i]] * coefficients[i]) = rhs`
        post_processing_func: A function that post-processes an optimization
            result appropriately (i.e., according to `round-trip`
            transformations).
        batch_initial_conditions: A tensor to specify the initial conditions. Set
            this if you do not want to use default initialization strategy.
        sequential: If False, uses joint optimization, otherwise uses sequential
            optimization.
    Returns:
        A two-element tuple containing
        - a `q x d`-dim tensor of generated candidates.
        - an associated acquisition value.
    """
    if not fixed_features_list:
        raise ValueError("fixed_features_list must be non-empty.")
    ff_candidate_list, ff_acq_value_list = [], []
    for fixed_features in fixed_features_list:
        candidate, acq_value = optimize_acqf(
            acq_function=acq_function,
            bounds=bounds,
            q=q,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            options=options or {},
            inequality_constraints=inequality_constraints,
            equality_constraints=equality_constraints,
            fixed_features=fixed_features,
            post_processing_func=post_processing_func,
            batch_initial_conditions=batch_initial_conditions,
            return_best_only=True,
            sequential=False,
        )
        ff_candidate_list.append(candidate)
        ff_acq_value_list.append(acq_value)
    ff_acq_values = torch.stack(ff_acq_value_list)
    best = torch.argmax(ff_acq_values)
    return ff_candidate_list[best], ff_acq_values[best]

#%% Multi-Information Source Problem

import os
import torch
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize, standardize
from botorch.utils.sampling import draw_sobol_samples
from botorch import fit_gpytorch_model
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.optimize import optimize_acqf
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
# from botorch.optim.optimize import optimize_acqf_mixed

#Training and Testing Data
# N = 25
# n = 100
# X = torch.rand(N,2)
# x = torch.zeros((n,3))
# x[:,0] = torch.linspace(0,1,n)
# fidelities = torch.tensor([0,1,2])
# X_fids = fidelities[torch.randint(3,(1,N))]
# X = torch.cat((X.T,X_fids)).T
# Y = get_y_fid(X)
# y = get_y_fid(x)

def generate_initial_data(N = 25):
    X = torch.rand(N,2)
    fidelities = torch.tensor([0,1,2])
    X_fids = fidelities[torch.randint(3,(1,N))]
    X = torch.cat((X.T,X_fids)).T
    Y = get_y_fid(X)
    return X, Y
    
def initialize_model(train_x, train_obj):
    model = SingleTaskMultiFidelityGP(
        train_x, 
        train_obj, 
        outcome_transform=Standardize(m=1),
        data_fidelity=2)   
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

def project(X):
    target_fidelities = {2: 0}
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

def get_mfkg(model):
    
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,
        columns=[2],
        values=[0])
    
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=10,
        raw_samples=1024,
        options={"batch_limit": 10, "maxiter": 200})
        
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project)

def optimize_mfkg_and_get_observation(mfkg_acqf):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""
    
    X_init = gen_one_shot_kg_initial_conditions(
        acq_function = mfkg_acqf,
        bounds=bounds,
        q=2,
        num_restarts=10,
        raw_samples=512)
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{2: 0}, {2: 1}, {2: 2}],
        q=2,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200})
    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = get_y_fid(new_x)
    return new_x, new_obj, cost

# with torch.no_grad():
#     y_bo = model.posterior(x).mean.cpu().numpy()
    
#Plot Stuff
# plt.figure()
# plt.plot(X[X[:,-1] == 0,0],Y[X[:,-1] == 0],'rs')
# plt.plot(X[X[:,-1] == 1,0],Y[X[:,-1] == 1],'bs')
# plt.plot(X[X[:,-1] == 2,0],Y[X[:,-1] == 2],'gs')
# plt.xlabel('x')
# plt.ylabel('y')

X,Y = generate_initial_data(N = 25)

bounds = torch.tensor([[0.0] * 3, [1.0] * 3])
bounds[-1,-1] = 2

cost_model = AffineFidelityCostModel(fidelity_weights={2: 0}, fixed_cost=5.0)
cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

cumulative_cost = 0.0
N_ITER = 2
NUM_RESTARTS = 10
RAW_SAMPLES = 512

for _ in range(N_ITER):
    mll, model = initialize_model(X,Y)
    fit_gpytorch_model(mll)
    mfkg_acqf = get_mfkg(model)
    new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
    X = torch.cat([X, new_x])
    Y = torch.cat([Y, new_obj])
    cumulative_cost += cost