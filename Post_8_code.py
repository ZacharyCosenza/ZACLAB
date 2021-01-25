# Zachary Cosenza
# Post 8 Code

#%% Part 1 Getting Outputs from BOtorch

import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_model

def get_Y(X):
    n,p = X.shape
    Y = X[:,0]**2 - X[:,1]**2
    return Y.reshape(-1,1)

def PlotGPContour(X,Y,params,key):
    l = 20
    x1 = np.linspace(0,1,l)
    x2 = x1
    C = np.zeros([l,l])
    for i in np.arange(l):
        for j in np.arange(l):
            x = np.array([x1[i],x2[j]]).reshape(1,2)
            if key == 'bo1':
                with torch.no_grad():
                    bo = params.posterior(torch.from_numpy(x))
                    C[i,j] = bo.mean.cpu().numpy()
            elif key == 'bo2':
                C[i,j] = params(torch.tensor(x.reshape(1,1,2)))
            elif key == 'real':
                C[i,j] = get_Y(x)
    return C

def MakePlot(X,Y,params,key):
    l = 20
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

#Make Plot
plt.figure(1)
MakePlot(X,Y,[],'real')

#Make BOtorch Model
train_X = torch.from_numpy(X)
train_Y = torch.from_numpy(Y)
model = SingleTaskGP(train_X=train_X, train_Y=train_Y)
mll = ExactMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)
model.eval() # set model (and likelihood)

#Plot BOtorch Results
plt.figure(2)
MakePlot(X,Y,model,'bo1') #Plot Contour

#%% Part 2 Getting Outputs from BOtorch and Customs

from botorch import fit_gpytorch_model
from botorch.models import FixedNoiseGP, ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.models.model import Model
from torch import Tensor
from botorch.utils import t_batch_mode_transform
from botorch.acquisition import AnalyticAcquisitionFunction

class Custom_Aq(AnalyticAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        maximize: bool = True) -> None:
        # we use the AcquisitionFunction constructor, since that of 
        # AnalyticAcquisitionFunction performs some validity checks that we don't want here
        super(AnalyticAcquisitionFunction, self).__init__(model)
        self.maximize = maximize

    @t_batch_mode_transform(expected_q=1)
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate the Upper Confidence Bound on the candidate set X using scalarization

        Args:
            X: A `(b) x d`-dim Tensor of `(b)` t-batches of `d`-dim design
                points each.

        Returns:
            A `(b)`-dim Tensor of Upper Confidence Bound values at the given
                design points `X`.
        """
        posterior = self.model.posterior(X)
        means = posterior.mean.squeeze(dim=-2)  # b x o
        return means

#Make BOtorch Model using Custom Acq
train_X = torch.from_numpy(X)
Y = get_Y(X)
train_Y = torch.from_numpy(Y)

NOISE_SE = 0.25
train_yvar = torch.tensor(NOISE_SE**2)
model_obj = FixedNoiseGP(train_X, train_Y, train_yvar.expand_as(train_Y))

model = ModelListGP(model_obj)
mll = SumMarginalLogLikelihood(model.likelihood, model)
fit_gpytorch_model(mll)
Custom = Custom_Aq(model)

#Plot BOtorch Results
plt.figure(3)
MakePlot(X,Y,Custom,'bo2') #Plot Contour

#%% Part 3 Find the Best Objective Value Point

from botorch.gen import gen_candidates_scipy

num_multi_starts = 20
candidates = []
acq_values = []
bounds = torch.tensor([[0,0],[1,1]])
for i in range(num_multi_starts):
    xo = torch.tensor(np.random.uniform(low = 0, high = 1, size = 2)).double()
    batch_candidates, batch_acq_values = gen_candidates_scipy(
            initial_conditions=xo,
            acquisition_function=Custom,
            lower_bounds=bounds[0],
            upper_bounds=bounds[1])
    candidates.append(batch_candidates)
    acq_values.append(batch_acq_values)

ind_acq = np.argmax(acq_values)
candidate_best = candidates[ind_acq]
plt.figure(3)
plt.plot(candidate_best[0][1],candidate_best[0][0],'rs')