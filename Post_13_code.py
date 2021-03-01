# Zachary Cosenza
# Post 13
# Multiobjective Optimization

import torch
from botorch.test_functions.multi_objective import BraninCurrin
import numpy as np
import random
import matplotlib.pyplot as plt

# Define Problem We Will Solve
def toy_problem(x):
    problem = BraninCurrin(negate=True)
    return torch.Tensor.numpy(problem(torch.tensor(x)))
problem = BraninCurrin(negate=True)

# Plot the Toy Function
x = np.random.uniform(0,1,size = (2000,2))
y = toy_problem(x)

plt.figure()
plt.plot(y[:,0],y[:,1],'k.',markersize = 1)
plt.xlabel('y_1')
plt.ylabel('y_2')

# 1-D Versions of Multiobjective Problems

# Part 1: Single Output Method

alphas = np.array([0,0.25,0.5,0.75,1]) #weight parameter

y_1_max,y_1_min = max(y[:,0]),min(y[:,0])
y_2_max,y_2_min = max(y[:,1]),min(y[:,1])

l = 20
xx, yy = np.meshgrid(np.linspace(0, 1, l), np.linspace(0, 1, l), sparse=True)
for k in range(len(alphas)):
    x1 = np.linspace(0,1,l)
    x2 = x1
    C = np.zeros([l,l])
    for i in np.arange(l):
        for j in np.arange(l):
            x = np.array([x1[i],x2[j]]).reshape(1,2)
            y = toy_problem(x)
            y_1_hat = (y[0][0] - y_1_min) / (y_1_max - y_1_min)
            y_2_hat = (y[0][1] - y_2_min) / (y_2_max - y_2_min)
            C[i,j] = alphas[k] * y_1_hat + (1 - alphas[k]) * y_2_hat
    plt.figure()
    plt.contourf(np.linspace(0, 1, l),np.linspace(0, 1, l),C)
    plt.colorbar()
    plt.xlabel('x_1')
    plt.ylabel('x_2')

# Plot Just Y1 and Y2

x1 = np.linspace(0,1,l)
x2 = x1
C1 = np.zeros([l,l])
C2 = np.zeros([l,l])
for i in np.arange(l):
    for j in np.arange(l):
        x = np.array([x1[i],x2[j]]).reshape(1,2)
        y = toy_problem(x)
        C1[i,j] = y[0][0]
        C2[i,j] = y[0][1]
plt.figure()
plt.contourf(np.linspace(0, 1, l),np.linspace(0, 1, l),C1)
plt.colorbar()
plt.xlabel('x_1')
plt.ylabel('x_2')
plt.figure()
plt.contourf(np.linspace(0, 1, l),np.linspace(0, 1, l),C2)
plt.colorbar()
plt.xlabel('x_1')
plt.ylabel('x_2')

# Larger the Best Desirability Function

# 1-D Example of LTB Desirability Function

y_ = np.linspace(0,1,1000)
L = 0
U = 1
bound = np.array([0.1,0.9])
rs = [0.25,0.5,1,1.5,2]
d = np.zeros([1000,len(rs)])
for i in range(len(rs)):   
    d[:,i] = ((y_-L)/(U-L))**rs[i]
    d[y_ >= bound[1],i] = 1
    d[y_ <= bound[0],i] = 0

plt.figure()
plt.plot(y_,d[:,0],'r')
plt.plot(y_,d[:,1],'r--')
plt.plot(y_,d[:,2],'k')
plt.plot(y_,d[:,3],'b--')
plt.plot(y_,d[:,4],'b')
plt.ylabel('d')
plt.xlabel('y')
plt.legend(['Bias Against 0','','','','Bias For 0'])

def get_LTB(y,bounds,r,w):
    d = np.zeros(len(bounds))
    for i in range(len(bounds)):
        if y[0][i] >= bounds[i][0]: #upper bound
            d[i] = 1
        elif y[0][i] <= bounds[i][1]: #lower bound
            d[i] = 0
        else:
            d[i] = (((y[0][i] - bounds[i][1]) / (bounds[i][0] - bounds[i][1]))**r[i])**w[i]
    D = np.prod(d)**(1/sum(w))
    return D

r = np.array([1,1]) #shape parameter
w = np.array([1,2]) #weight parameter
bounds = [(y_1_max,y_1_min),(y_2_max,y_2_min)]

l = 20
xx, yy = np.meshgrid(np.linspace(0, 1, l), np.linspace(0, 1, l), sparse=True)
x1 = np.linspace(0,1,l)
x2 = x1
C = np.zeros([l,l])
for i in np.arange(l):
    for j in np.arange(l):
        x = np.array([x1[i],x2[j]]).reshape(1,2)
        y = toy_problem(x)
        C[i,j] = get_LTB(y,bounds,r,w)
plt.figure()
plt.contourf(np.linspace(0, 1, l),np.linspace(0, 1, l),C)
plt.colorbar()
plt.xlabel('x_1')
plt.ylabel('x_2')

#%% NSGA-II

def parato_rank(Y):
    Y = np.array(Y)
    #Inputs: X and Y
    #Outputs: Number of domination per point
    n,y_dim = Y.shape
    #Get Number of Dominations per Point
    num_dom = np.zeros(n)
    for i in range(n):
        num_dom[i] = np.sum(np.prod(Y[i,:] > Y,1))
    return num_dom

def get_F(num_dom):
    #Input: Number of dominations per point
    #Output: Partitions
    F = np.zeros(len(num_dom))
    unique_doms = np.unique(num_dom)
    i = len(unique_doms)
    for dom in unique_doms:
        ind_F = (num_dom == dom)
        F[ind_F] = i
        i = i - 1
    return F
        
def get_distance(Y):
    #Input: Y
    #Output: Distances vector neighbor distances
    n,p = Y.shape
    distances = np.zeros(n)
    for i in range(n):
        d = np.zeros(p)
        for j in range(p):
            ind = [x for k, x in enumerate(range(n)) if k != i]
            if not ind:
                print(ind)
            d[j] = min(np.abs(Y[i,j] - Y[ind,j]))
        distances[i] = sum(d)
    return distances

def get_NSGAII(funct,px,py,*args):
    #Inital Problem
    iter_max = 100
    N = 10
    x = np.random.uniform(low = 0, high = 1, size = (2*N,px))
    y = np.zeros([2*N,py])
    for i in range(2*N):
        y[i,:] = funct(x[i,:],*args)
    X_list = []
    Y_list = []
    for k in range(iter_max):
        # Pareto Rank of Population
        num_dom = parato_rank(y)
        F = get_F(num_dom)
        ind_pareto = np.argsort(F)
        
        # Get N Population Based on NSGA-II
        P = []
        for i in range(int(max(F))):
            next_group = ind_pareto[F == i + 1]
            if len(P) + len(next_group) <= N:
                # Add to List Group-ally
                P = np.hstack((P,next_group))
            else:
                # Add to List Individually
                length_needed = N - len(P)
                if length_needed > 1:
                    distances = get_distance(y[next_group,:])
                    ind_distances = np.argsort(-distances)
                    next_group_distance = next_group[ind_distances[:length_needed]]
                    P = np.hstack((P,next_group_distance))
                else:
                    P = np.hstack((P,next_group))
                break
        P = np.array(P,dtype = int)
        
        #Tournament Selection of Probablistic Best Points (k-Point)
        distances = get_distance(y[P,:])
        distances[distances == np.inf] = max(distances[np.isfinite(distances)])
        p_distances = distances / sum(distances)
        parents = random.choices(P,p_distances,k = N)
        x_parents = x[parents,:]
        
        # Crossover
        num_children = N
        x_children = np.zeros([num_children,px])
        for i in range(num_children):
            #Get Two Parents
            ind_parents = np.random.choice(np.arange(N), 2, replace = False)
            ind_parents = np.array(ind_parents,dtype = int)
            x_couple = x_parents[ind_parents,:]
            #Cut Pointx_
            cut = np.random.randint(low = 0, high = px)
            #Crossover
            x_children[i,:] = np.concatenate((x_couple[0,0:cut],x_couple[1,cut:px]))
        
        # Mutation
        p_mut = 0.1
        for i in range(N):
            for j in range(px):
                r = np.random.uniform(low = 0, high = 1)
                if r <= p_mut:
                    x_parents[i,j] = np.random.uniform(low = 0, high = 1)
                    
        #Combination
        x = np.concatenate((x_parents,x_children))
        
        y = np.zeros([2*N,py])
        y_parents = np.zeros([N,py])
        for i in range(2*N):
            y[i,:] = funct(x[i,:],*args)
        for i in range(N):
            y_parents[i,:] = funct(x_parents[i,:],*args)
            
        X_list.append(x_parents)
        Y_list.append(y_parents)
        
    return X_list,Y_list

px = 2
py = 2
X,Y = get_NSGAII(toy_problem,px,py)

plt.figure()
for i in range(len(Y)):
    plt.plot(Y[i][:,0],Y[i][:,1],'k.',markersize=1)
plt.xlabel('y_1')
plt.ylabel('y_2')
plt.title('Pareto Optimization using NSGA-II')

#%% Botorch Implementation of Multiobjective Optimization

from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch.utils.transforms import unnormalize
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.multi_objective.box_decomposition import NondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.sampling import sample_simplex
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.utils.sampling import draw_sobol_samples

import warnings
warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Generate Data from Sobol Samples and Define Problem
N = 10
train_x = draw_sobol_samples(bounds=problem.bounds,n = 1, q = N, 
                             seed=torch.randint(1000000, (1,)).item()).squeeze(0)
train_obj = problem(train_x)
x_ref = [-18,-6]
plt.figure()
plt.plot(train_obj[:,0],train_obj[:,1],'k.')

# Get Model and Define Log Likelihood
model = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=train_obj.shape[-1]))
mll = ExactMarginalLogLikelihood(model.likelihood, model)

# Train Hyperparameters
mll = fit_gpytorch_model(mll)

# Define Multiobjective Model
partitioning = NondominatedPartitioning(num_outcomes=train_obj.shape[-1], Y=train_obj)
qehvi_sampler = SobolQMCNormalSampler(num_samples=1000)
acq_func = qExpectedHypervolumeImprovement(
        model=model,
        ref_point=x_ref,  # use known reference point 
        partitioning=partitioning,
        sampler=qehvi_sampler)

# Optimize
BATCH_SIZE = 2
candidates, _ = optimize_acqf(
    acq_function=acq_func,
    bounds=problem.bounds,
    q=BATCH_SIZE,
    num_restarts=20,
    raw_samples=1024,  # used for intialization heuristic
    options={"batch_limit": 5, "maxiter": 200, "nonnegative": True},
    sequential=True)

new_x =  unnormalize(candidates.detach(), bounds=problem.bounds)
new_obj = problem(new_x)
plt.plot(new_obj[:,0],new_obj[:,1],'r.')
plt.xlabel('f1')
plt.ylabel('f2')
plt.legend(['Training Data','Expected Optima'])
plt.title('Expected Hypervolume Improvement using GPs')

# Calculate Hypervolume of New Points
pareto_mask = is_non_dominated(new_obj)
pareto_y = new_obj[pareto_mask]
hv = Hypervolume(ref_point=problem.ref_point)
volume = hv.compute(pareto_y)