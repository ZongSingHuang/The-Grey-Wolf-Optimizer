# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 08:18:08 2021

@author: zongsing.huang
"""


import numpy as np
import matplotlib.pyplot as plt

def fitness(X):
    # Rosenbrock
    if X.ndim==1:
        X = X.reshape(1, -1)
    
    left = X[:, :-1].copy()
    right = X[:, 1:].copy()
    F = np.sum(100*(right - left**2)**2 + (left-1)**2, axis=1)
    
    return F

#%% 參數設定
P = 30
D = 30
G = 500
T = 100
ub = 30*np.ones([P, D])
lb = -30*np.ones([P, D])
a_max = 2
statistical_experiment = np.zeros(T)
loss_curve = np.zeros([T, G])

for t in range(T):
    #%% 初始化
    X = np.random.uniform(low=lb, high=ub, size=[P, D])
    X_alpha = np.zeros(D)
    F_alpha = np.inf
    X_beta = np.zeros_like(X_alpha)
    F_beta = np.inf
    X_delta = np.zeros_like(X_alpha)
    F_delta = np.inf
    
    #%% 迭代
    for g in range(G):
        # Step1. 適應值計算
        F = fitness(X)
        
        # Step2. 更新alpha, beta, delta
        for i in range(P):
            # Step2-1. 更新F_alpha
            if F[i]<F_alpha:
                X_alpha = X[i]
                F_alpha = F[i]
            # Step2-2. 更新F_beta
            if F[i]<F_beta:
                X_beta = X[i]
                F_beta = F[i]
            # Step2-3. 更新F_delta
            if F[i]<F_delta:
                X_delta = X[i]
                F_delta = F[i]
        loss_curve[t, g] = F_alpha
        statistical_experiment[t] = F_alpha
        
        # Step3. 更新a
        a = a_max*(1 - g/G)
        
        # Step4. 更新alpha, beta, delta
        # Step4-1. 更新X_alpha
        r1 = np.random.uniform(size=[P, D])
        r2 = np.random.uniform(size=[P, D])
        A_alpha = 2*a*r1 - a
        C_alpha = 2*r2
        D_alpha = np.abs(C_alpha*X_alpha-X)
        X1 = X_alpha - A_alpha*D_alpha
        # Step4-2. 更新X_beta
        r1 = np.random.uniform(size=[P, D])
        r2 = np.random.uniform(size=[P, D])
        A_beta = 2*a*r1 - a
        C_beta = 2*r2
        D_beta = np.abs(C_beta*X_beta-X)
        X2 = X_beta - A_beta*D_beta
        # Step4-3. 更新X_delta
        r1 = np.random.uniform(size=[P, D])
        r2 = np.random.uniform(size=[P, D])
        A_delta = 2*a*r1 - a
        C_delta = 2*r2
        D_delta = np.abs(C_delta*X_delta-X)
        X3 = X_delta - A_delta*D_delta
        
        # Step5. 更新X
        X = (X1+X2+X3)/3
    
#%% 作畫
plt.figure()
plt.title('Original')
plt.plot(loss_curve.mean(axis=0))
plt.grid()
plt.xlabel('Iteration')
plt.ylabel('Fitness')

#%% 統計分析
mean = statistical_experiment.mean()
std = statistical_experiment.std()
print(mean)
print(std)