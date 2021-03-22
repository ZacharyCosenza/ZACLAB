# Zachary Cosenza
# Post 17 Linear Basis Functions in Bayesian Regression

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def f(X, noise_variance):
    return -0.3 + 0.5 * X + noise(X.shape, noise_variance)
    
def g(X, noise_variance):
    return 0.5 + np.sin(2 * np.pi * X) + noise(X.shape, noise_variance)

def noise(size, variance):
    return np.random.normal(scale=np.sqrt(variance), size=size)

def expand(x, bf, bf_args=None):
    if bf_args is None:
        return np.concatenate([np.ones(x.shape), bf(x)], axis=1)
    else:
        return np.concatenate([np.ones(x.shape)] + [bf(x, bf_arg) for bf_arg in bf_args], axis=1)

def identity_basis_function(x):
    return x

def gaussian_basis_function(x, sigma=0.1):
    mu = 2
    return np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)

def polyomial_basis_function(x, power = 2):
    return x ** power

def posterior(Phi, t, alpha, beta):
    S_N_inv = alpha * np.eye(Phi.shape[1]) + beta * Phi.T.dot(Phi)
    S_N = np.linalg.inv(S_N_inv)
    m_N = beta * S_N.dot(Phi.T).dot(t)
    return m_N, S_N

def posterior_predictive(Phi_test, m_N, S_N, beta):
    """Computes mean and variances of the posterior predictive distribution."""
    y = Phi_test.dot(m_N)
    # Only compute variances (diagonal elements of covariance matrix)
    y_var = 1 / beta + np.sum(Phi_test.dot(S_N) * Phi_test, axis=1)
    return y, y_var

def get_plots(X,Y,alpha,beta):
    #Make Posterior Plot for Weights
    plt.subplot(1,3,1)
    w1_list = np.linspace(-1,1,100)
    w0_list = np.linspace(-1,1,100)
    Phi = expand(X,identity_basis_function)
    m_N,S_N = posterior(Phi,Y,alpha,beta)
    post_plot = np.zeros([len(w0_list),len(w1_list)])
    for i in np.arange(len(w0_list)):
        for j in np.arange(len(w1_list)):
            post_plot[i,j] = multivariate_normal.pdf(np.array([w0_list[i],w1_list[j]]),
                                                      mean = m_N.ravel(),
                                                      cov = S_N)
    plt.pcolormesh(w1_list,w0_list,post_plot)
    plt.xlabel('w1')
    plt.ylabel('w0')
    # Collect Samples of w0 and w1 and Plot
    plt.subplot(1,3,2)
    N_samples = 10
    w_samples = np.random.multivariate_normal(m_N.ravel(), S_N, N_samples).T
    x_test = np.linspace(-1,1,100).reshape(-1, 1)
    Phi_test = expand(x_test,identity_basis_function)
    y_test = Phi_test.dot(w_samples)
    plt.plot(x_test,y_test,'k')
    plt.xlabel('x')
    plt.ylabel('y')
    #Male MLE and Variance Plot
    plt.subplot(1,3,3)
    y_mle = Phi_test.dot(m_N)
    std_mle = np.sqrt(1/beta + np.sum(Phi_test.dot(S_N) * Phi_test, axis=1))
    plt.plot(x_test,y_mle,'k')
    plt.plot(x_test,y_mle+std_mle.reshape(-1,1),'k--')
    plt.plot(x_test,y_mle-std_mle.reshape(-1,1),'k--')
    plt.plot(X,Y,'rs')
    plt.xlabel('x')
    plt.ylabel('y')
    

# Training dataset sizes
N = 20
beta = 25.0
alpha = 2.0
X = np.random.rand(N, 1) * 2 - 1
Y = f(X, noise_variance=1/beta)

# Test observations
X_test = np.linspace(-1, 1, 100).reshape(-1, 1)
y_true = f(X_test, noise_variance=0)
    
# Design matrix of test observations
Phi_test = expand(X_test,identity_basis_function)

# Design matrix of training observations
Phi_train = expand(X,identity_basis_function)

# Mean and covariance matrix of posterior
m_N, S_N = posterior(Phi_train,Y,alpha,beta)

# Mean and variances of posterior predictive 
y, y_var = posterior_predictive(Phi_test, m_N, S_N, beta)

get_plots(X,Y,alpha,beta)