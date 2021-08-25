# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 12:18:43 2020

@author: bryan
"""


import math
import numpy as np
import matplotlib.pyplot as plt
import random
from fractions import Fraction

C=16            #from dist of C the dilation and asymmetry control
n=C // 4        # #patterns
N=5*C           # #neurons
ti=10           # #of time steps of dynamics
t=10            # #of time steps of simulation
m0=0.8

beta=np.arange(0.01,0.81, 0.01)

ave=200
def f(x,y):
        return(sum(((C**K)*math.exp(-C)/math.factorial(K))*sum(sum((((1+x)**(K-p)*(1-x)**p)/(2**(K*n)))*Fraction(math.factorial(K),math.factorial(p)*math.factorial(K-p))*Fraction(math.factorial(K*(n-1)),math.factorial(s)*math.factorial(K*(n-1)-s))*np.tanh((K*n-2*p-2*s)*y) for s in range(0,K*(n-1)+1))for p in range(0,K+1)) for K in range(0,N+1)))


mfin=np.zeros(len(beta))
m=np.zeros(shape=ti)
m[0]=m0
for b in range(len(beta)): 
    for j in range(1,ti): 
        m[j]=f(m[j-1],beta[b])
    mfin[b]=m[ti-1]
print(mfin)
c=np.zeros(shape=(N,N))
for i in range(N):
    for j in range(N):
        elements = [1,0]
        probabilities = [Fraction(C,N),1-Fraction(C,N)]
        c[i][j]=np.random.choice(elements, 1, p=probabilities)   #dilation and asymmetry control

V=np.zeros(shape=(N,n))                                          #random pattern generation
for i in range(N):
    states = [1,-1]
    prob = [0.5,0.5]
    V[i]=np.array(np.random.choice(states, n, p=prob))    

w=np.zeros(shape=(N,N))
for i in range(N):
    for j in range(N):
        w[i][j]=c[i][j]*sum(V[i][p]*V[j][p] for p in range(n))
    w[i][i]=0

 
S0=np.zeros(shape=(N,len(beta),ave))
q=random.sample(range(1, N), N // 10)
z=np.zeros(shape=(N,ave)) 
for b in range(len(beta)):
    for a in range(ave):
        for i in range(N):
            if i in q:
                S0[i][b][a]=(-1)*(V[i][0])
            else:
                S0[i][b][a]=V[i][0]              
            z[i][a]=np.random.normal(0,1,1)
            
S=np.zeros(shape=(N, len(beta), ave, t))
for b in range(len(beta)):
    for a in range(ave):
        for k in range (1,t):
            for i in range (N):
                S[i][b][a][0]=S0[i][b][a]
                if sum(w[i][j]*S[j][b][a][k-1] for j in range(N))+z[i][a]/beta[b]>0:
                    S[i][b][a][k]=1
                elif sum(w[i][j]*S[j][b][a][k-1] for j in range(N))+z[i][a]/beta[b]<0:
                    S[i][b][a][k]=-1
                else:
                    S[i][b][a][k]=S[i][b][a][k-1]

m=np.zeros(shape=(n,len(beta),ave,t))
m1=np.zeros(shape=(N,n,len(beta),ave))
for b in range(len(beta)):
    for a in range(ave):
        for l in range(ti):
            for j in range(n):
                for i in range(N):
                    if S[i][b][a][l]==V[i][j]:
                        m1[i][j][b][a]=1
                    else:
                        m1[i][j][b][a]=-1
                m[j][b][a][l]=sum(m1[k][j][b][a] for k in range(N))
    

mcorrect=np.zeros(shape=(len(beta),ave,t))
mwrong=np.zeros(shape=(len(beta),ave,t))
mcorrectfin=np.zeros(shape=(len(beta),ave))
mwrongfin=np.zeros(shape=(len(beta),ave))
mcorrectfinal=np.zeros(len(beta))
mwrongfinal=np.zeros(len(beta))
for b in range(len(beta)):
    for a in range (ave):
        for l in range(1,t):
            mcorrect[b][a][l]=m[0][b][a][l]/N
            mwrong[b][a][l]=m[1][b][a][l]/N
        mcorrectfin[b][a]=mcorrect[b][a][t-1]
        mwrongfin[b][a]=mwrong[b][a][t-1]
    mcorrectfinal[b]=sum(mcorrectfin[b][d] for d in range(ave))/ave
    mwrongfinal[b]=sum(mwrongfin[b][d] for d in range(ave))/ave
print(mcorrectfinal)

f = plt.figure()
f, ax = plt.subplots()    
plt.plot(beta,mcorrectfinal,label="correct",marker='+', color = 'red')
plt.plot(beta,mfin,label=f"dynamics, N={N}",linestyle = '-',color = 'green')
ax.legend(loc =2, prop={'size': 12}) 
plt.xlabel(r'$beta$', fontsize = 20)
plt.ylabel(r'$m$', fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=15)

g = plt.figure()
g, ax = plt.subplots()    
plt.plot(beta,mwrongfinal,label="incorrect",marker='x', color = 'blue')
plt.plot(beta,mfin,label=f"dynamics, N={N}",linestyle = '-',color = 'green')
ax.legend(loc =2, prop={'size': 12}) 
plt.xlabel(r'$beta$', fontsize = 20)
plt.ylabel(r'$m$', fontsize = 20)
plt.tick_params(axis='both', which='major', labelsize=15)
