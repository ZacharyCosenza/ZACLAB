# Zachary Cosenza
# Post 7 Trust Region Optimization

#%% Part 1 Trust Region Algorithms and Optimization

import numpy as np
import matplotlib.pyplot as plt

def get_himmel(X): 
    x = X[0]
    y = X[1]
    
    #2-D Himmelblau Function
    f = (x**2 + y - 11)**2 + (x + y**2 - 7)**2
    
    #2-D Himmelblau Function Derivative
    dfx = 4*x**3+4*x*y-44*x+2*x+2*y**2-14
    dfy = 2*x**2+2*y-22+4*x*y+4*y**3-28*y
    df = np.hstack((dfx,dfy))
    
    #2-D Himmelblau Function Second Derivative
    dfxx = 12*x**2+4*y-44+2
    dfxy = 4*x+4*y
    dfyx = dfxy
    dfyy = 2+4*x+12*y**2-28
    ddf = np.hstack((np.vstack((dfxx,dfyx)),np.vstack((dfxy,dfyy))))
    return f, df, ddf

def PlotContour(x,key):
    l = len(x)
    x1 = x
    x2 = x1
    C = np.zeros([l,l])
    for i in np.arange(l):
        for j in np.arange(l):
            x = np.array([x1[i],x2[j]])
            if key == 'himmel':
                f,df,ddf = get_himmel(x)
                C[i,j] = f
    return C

def MakePlot(key):
    l = 30
    x = np.linspace(-5,5,l)
    xx, yy = np.meshgrid(x,x, sparse=True)
    C = PlotContour(x,key)
    plt.contourf(x,x,C)
    plt.colorbar()

def get_m(x,p):
    f,df,ddf = get_himmel(x)
    p = np.array(p)
    m = f + df @ p + 0.5 * p @ ddf @ p
    return m

def get_p(x):
    f,df,ddf = get_himmel(x)
    p = -np.linalg.pinv(ddf) @ df
    return p

def get_rho(x,p):
    fk,_,_ = get_himmel(x)
    fkk,_,_ = get_himmel(x+p)
    m0 = get_m(x,np.zeros(len(x)))
    mp = get_m(x,p)
    rho = (fk-fkk)/(m0-mp)
    return rho

def projection(x):
    if x[0] < -5:
        x[0] = -5
    elif x[0] > 5:
        x[0] = 5
    if x[1] < -5:
        x[1] = -5
    elif x[1] > 5:
        x[1] = 5
    return x

def trust_region(p,x,delta,k):
    eta = 0.1
    delta_hat = 1
    rho = get_rho(x[k,:],p)
    if rho < 0.25:
        delta = 0.25*delta
    else:
        if rho > 0.75 and np.linalg.norm(p,2) < delta:
            delta = np.min(np.array([2*delta,delta_hat]))
        else:
            delta = delta
    if rho > eta or rho <= 0:
        x[k+1,:] = x[k,:] + p
    else:
        x[k+1,:] = x[k,:]
    x[k+1,:] = projection(x[k+1,:])
    return x,delta

plt.figure(figsize=(20,10))
plt.subplot(1,2,1)
MakePlot('himmel')

K = 10
alpha = 0.01

#Initalize
rho = np.zeros(K)
fnm_record = np.zeros(K)
fgd_record = np.zeros(K)
fc_record = np.zeros(K)
ffull_record = np.zeros(K)
fdl_record = np.zeros(K)
x_full = np.zeros([K,2])
x_c = np.zeros([K,2])
x_dl = np.zeros([K,2])
xnm = np.zeros([K,2])
xgd = np.zeros([K,2])

start = np.array([-4,-4])
x_full[0,:] = start
x_c[0,:] = start
x_dl[0,:] = start
xnm[0,:] = start
xgd[0,:] = start
x_full[0,:] = np.random.uniform(low = -5,high = 5, size = 2)
x_c[0,:] = np.random.uniform(low = -5,high = 5, size = 2)
x_dl[0,:] = np.random.uniform(low = -5,high = 5, size = 2)
xnm[0,:] = np.random.uniform(low = -5,high = 5, size = 2)
xgd[0,:] = np.random.uniform(low = -5,high = 5, size = 2)
delta_full = 0.1
delta_c = 0.1
delta_dl = 0.1

#Trust Region from SciPy
from scipy.optimize import minimize

#Make Wrappers of BFGS for scipy optimize minimize
def wrapper_f(p):
    y,_,_ = get_himmel(p)
    return y

def wrapper_df(p):
    _,dy,_ = get_himmel(p)
    return dy

def wrapper_ddf(p):
    _,_,ddf = get_himmel(p)
    return ddf

dogleg_info = minimize(wrapper_f,x0 = start, method='dogleg', 
                       jac = wrapper_df, hess = wrapper_ddf)
x_scipy_opt = dogleg_info['x']

for k in np.arange(K-1):
    
    #Trust Region Method Full Step
    p_full = get_p(x_full[k,:])
    x_full,delta_full = trust_region(p_full,x_full,delta_full,k)
    f,_,_ = get_himmel(x_full[k+1,:])
    ffull_record[k] = f
    
    #Trust Region Cauchy Point
    _,g,B = get_himmel(x_c[k,:])
    v = g @ B @ g.T
    if v <= 0:
        tau = 1
    else:
        tau = np.min([np.linalg.norm(g,2)**3/(delta_c * v),1])
    p_c = -tau * delta_c / np.linalg.norm(g,2) * g
    x_c,delta_c = trust_region(p_c,x_c,delta_c,k)    
    f,_,_ = get_himmel(x_c[k+1,:])
    fc_record[k] = f
    
    #DogLeg Method
    _,g,B = get_himmel(x_dl[k,:])
    p_u = - g.T @ g / (g.T @ B @ g) * g
    p_B = - np.linalg.pinv(B) @ g
    if tau > 0 and tau <= 1:
        p_dl = tau * p_u
    elif tau > 1 and tau <= 2:
        p_dl = p_u + (tau - 1) * (p_B - p_u)
    x_dl,delta_dl = trust_region(p_dl,x_dl,delta_dl,k)    
    f,_,_ = get_himmel(x_dl[k+1,:])
    fdl_record[k] = f
    x_dl[k+1,:] = projection(x_dl[k+1,:])
            
    #Comparison to Newton's Method
    pnm = get_p(xnm[k,:])
    xnm[k+1,:] = xnm[k,:] + alpha * pnm
    ff,_,_ = get_himmel(xnm[k+1,:])
    fnm_record[k] = ff
    xnm[k+1,:] = projection(xnm[k+1,:])
    
    #Comparison to Gradient Descent
    _,pgd,_ = get_himmel(xgd[k,:])
    xgd[k+1,:] = xgd[k,:] - alpha * pgd
    fff,_,_ = get_himmel(xgd[k+1,:])
    fgd_record[k] = fff
    xgd[k+1,:] = projection(xgd[k+1,:])

plt.plot(x_full[:,0],x_full[:,1],'c.')
plt.plot(x_c[:,0],x_c[:,1],'g.')
plt.plot(x_dl[:,0],x_dl[:,1],'y.')
plt.plot(xgd[:,0],xgd[:,1],'r.')
plt.plot(xnm[:,0],xnm[:,1],'b.')
plt.plot(x_scipy_opt[0],x_scipy_opt[1],'y*')
plt.xlabel('x_1',fontsize=20)
plt.ylabel('x_2',fontsize=20)

plt.subplot(1,2,2)
plt.plot(np.arange(len(ffull_record)-1),ffull_record[0:-1],'c',linewidth=6)
plt.plot(np.arange(len(fc_record)-1),fc_record[0:-1],'g',linewidth=6)
plt.plot(np.arange(len(fdl_record)-1),fdl_record[0:-1],'y',linewidth=6)
plt.plot(np.arange(len(fnm_record)-1),fnm_record[0:-1],'b',linewidth=6)
plt.plot(np.arange(len(fgd_record)-1),fgd_record[0:-1],'r',linewidth=6)
plt.legend(['Full Step','Cauchy Point','Dog-Leg','Newton''s Method','Gradient Descent'],
           fontsize=20)

#%% Part 2 Positive Definitness of f(x)

def is_pos_def(A):
    return np.all(np.linalg.eigvals(A) > 0)

x = np.linspace(-5,5,100)
l = len(x)
x1 = x
x2 = x1
C = np.zeros([l,l])
for i in np.arange(l):
    for j in np.arange(l):
        x = np.array([x1[i],x2[j]])
        _,_,ddf = get_himmel(x)
        C[i,j] = is_pos_def(ddf)

X, Y = np.meshgrid(x1, x2)
plt.figure(figsize=(10,10))
plt.contourf(X, Y, C, levels=[0,0.9])
plt.xlabel('x_1',fontsize=20)
plt.ylabel('x_2',fontsize=20)