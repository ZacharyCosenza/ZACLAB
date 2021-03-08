# Zachary Cosenza
# Post 15 Custom GPYtorch Kernel

#%% Part 1 Regular Kernel Functions

import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
from gpytorch.constraints import Positive

import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 100)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)

# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()
    
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    test_x = torch.linspace(0, 1, 51)
    observed_pred = likelihood(model(test_x))
    
with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Mean', 'Confidence'])

#%% Part 2 and 3 Custom Kernel and Hyperparameters

# Training data is 100 points in [0,1] inclusive regularly spaced
N = 20
p = 2
train_x = torch.zeros(size = (N,p))
train_x[:,0] = torch.linspace(0, 1, N)
train_x[0:10,1] = 1
# train_y = torch.sin(train_x[:,0] * (2 * math.pi)) + torch.randn(train_x.size(0)) * math.sqrt(0.1) 
train_y = train_x[:,0]

#Output 0
test_x1 = torch.zeros(size = (100,p))
test_x1[:,0] = torch.linspace(0, 1, 100)

#Output 1
test_x2 = torch.ones(size = (100,p))
test_x2[:,0] = torch.linspace(0, 1, 100)

# Wrap training, prediction and plotting from the ExactGP-Tutorial into a function,
# so that we do not have to repeat the code later on
def train(train_x,train_y,model,likelihood,training_iter=training_iter):
    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_x)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()

def predict(model,likelihood,test_x):
    model.eval()
    likelihood.eval()
    # Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        # Test points are regularly spaced along [0,1]
        return likelihood(model(test_x))

def plot(observed_pred,test_x,train_x):
    with torch.no_grad():
        # Initialize plot
        f, ax = plt.subplots(1, 1, figsize=(4, 3))
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()
        # Plot training data as black stars
        ax.plot(train_x[:,0].numpy(), train_y.numpy(), 'k*')
        # Plot predictive means as blue line
        ax.plot(test_x[:,0].numpy(), observed_pred.mean.numpy(), 'b.')
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x[:,0].numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
        ax.set_ylim([-3, 3])
        ax.legend(['Observed Data', 'Mean', 'Confidence'])

class CustomKernel(gpytorch.kernels.Kernel):
    def __init__(self):
        
        super().__init__()
        
        self.register_parameter(name='raw_length', 
                                parameter=torch.nn.Parameter(torch.zeros(2, 1)))
        self.register_parameter(name='raw_outputscale',
                                parameter=torch.nn.Parameter(torch.zeros(2, 1)))
        
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
        # diff = (x1 @ x2.T).div(self.length)
        # diff = torch.tensor(get_custom_covar(self,x1,x2)).float()
        diff = torch.tensor(get_custom_covar_IS(self,x1,x2)).float()
        diff.where(diff == 0, torch.as_tensor(1e-20))
        return diff
    
def get_custom_covar(self,x1,x2):
    x1 = x1.numpy()
    x2 = x2.numpy()
    n,p = x1.shape
    N,_ = x2.shape
    K = np.zeros([n,N])
    for i in np.arange(n):
        for j in np.arange(N):
            S = np.diag(self.length.detach().numpy().reshape(-1,))
            S_inv = np.linalg.pinv(S)**2
            X = (x1[i,:]-x2[j,:]).reshape(-1,1)
            r = X.T @ S_inv @ X
            K[i,j] = np.exp(-r)
    return K

def get_custom_covar_IS(self,x1,x2):
    x1 = x1.numpy()
    x2 = x2.numpy()
    n,p = x1.shape
    N,_ = x2.shape
    K = np.zeros([n,N])
    for i in np.arange(n):
        IS = int(x1[i,1])
        for j in np.arange(N):
            L0 = self.length.detach().numpy()[0]
            S = np.diag(L0)
            S_inv = np.linalg.pinv(S)**2
            X = (x1[i,0]-x2[j,0]).reshape(-1,1)
            r = X.T @ S_inv @ X
            K[i,j] = self.outputscale.detach().numpy()[0] * np.exp(-r)
            if IS != 0 and IS == x2[j,1]:
                L = self.length.detach().numpy()[1]
                S = np.diag(L)
                S_inv = np.linalg.pinv(S)**2
                r = X.T @ S_inv @ X
                K[i,j] = K[i,j] + self.outputscale.detach().numpy()[1] * np.exp(-r)
    return K

# Use the simplest form of GP model, exact inference
class CustomModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # base_covar_module = CustomKernel()
        # self.covar_module = gpytorch.kernels.ScaleKernel(base_covar_module)
        self.covar_module = CustomKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# initialize the new model
model = CustomModel(train_x, train_y, likelihood)

# set to training mode and train
model.train()
likelihood.train()
train(train_x,train_y,model, likelihood)

# Get into evaluation (predictive posterior) mode and predict
model.eval()
likelihood.eval()

observed_pred = predict(model,likelihood,test_x1)
plot(observed_pred,test_x1,train_x)

observed_pred = predict(model,likelihood,test_x2)
plot(observed_pred,test_x2,train_x)

#%% Part 4 Regular Kernel for Comparison

# Use the simplest form of GP model, exact inference
class RegularModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        base_covar_module = gpytorch.kernels.RBFKernel()
        self.covar_module = gpytorch.kernels.ScaleKernel(base_covar_module)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Clean Data for Comparison
train_x_comp = train_x[train_x[:,1] == 0,0]
train_y_comp = train_y[train_x[:,1] == 0]
test_x_comp = test_x1[:,0]

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()

# initialize the new model
model = RegularModel(train_x_comp, train_y_comp, likelihood)

# set to training mode and train
model.train()
likelihood.train()
train(train_x_comp,train_y_comp,model,likelihood)

# Get into evaluation (predictive posterior) mode and predict
model.eval()
likelihood.eval()

observed_pred = predict(model,likelihood,test_x_comp)
plot(observed_pred,test_x1,train_x)