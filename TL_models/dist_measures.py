"""
Various implementations of nonparametric distribution divergence measures 
"""

import numpy as np
import math 
from sklearn.metrics import  pairwise_distances
from sklearn.svm import SVC


def PAD(Xs,Xt,kernel='rbf',Ys=1,Yt=1):
    "Proxy A-distance: Analysis of representations for domain adaptation S Ben-David et al. (2007)"
    
    ns=Xs.shape[0]
    nt=Xt.shape[0]
    Xs=Xs[:nt,:]
    y=np.vstack((np.zeros((nt,1)),np.ones((nt,1))))
    X=np.vstack((Xs,Xt))
    svm = SVC(kernel=kernel,probability=True)
    svm.fit(X,np.ravel(y))
    error=1-svm.score(X,y)
    N=(Ys+Yt)/Yt
    A_dist=2*(1-N*error)
    return A_dist


def median_heuristic(X1, X2, type_flag='square'):
    "median heuristic for length scale selection for two-sample tests"
    if type_flag == 'square':
        'v = E([||x_i-x_j||] | 1 ≤ i < j ≤ n) from \"A Kernel Two-Sample Test, Gretton et al.\"'
        return 0.5*np.median(np.square(pairwise_distances(X1, X2)))**(1/2)
    elif type_flag == 'abs':
        'v =Sqrt(H_n); H_n = E([x_i-x_j]^2 | 1 ≤ i < j ≤ n) from \"Large sample analysis of the median heuristic, Garreau et al.\"'
        return np.median(pairwise_distances(X1, X2))

def rbf(X,Y,l):
    K=np.exp(-np.square(pairwise_distances(X, Y, metric='euclidean')) / (2*l**2))
    return K

def MMD(Xs,Xt, l=None, type_flag='square'):
    "The maximum mean discrepancy (MMD): A Kernel Two-Sample Test, Gretton et al."
    ns, nt = Xs.shape[0],  Xt.shape[0]
    if l==None:
        l = median_heuristic(Xs, Xt, type_flag=type_flag)

    K_ss2 = (ns**2)/(ns*(ns-1))*np.mean(rbf(Xs,Xs, l)-rbf(Xs,Xs, l)*np.eye(Xs.shape[0])) 
    K_tt2 = (nt**2)/(nt*(nt-1))*np.mean(rbf(Xt,Xt, l)-rbf(Xt,Xt, l)*np.eye(Xt.shape[0]))
    K_st2 = 2 / (ns*nt) * np.mean(rbf(Xs,Xt, l))
    return  K_ss2 + K_tt2 - K_st2

def JMMD(Xs, Xt, ys, yt, l=None):
    mmd = MMD(Xs, Xt)
    cS = np.unique(ys)
    cT = np.unique(yt)
    if l is None:
        D = np.square(pairwise_distances(Xs, Xt))
        l = 0.5*np.median(D)**(1/2)
    for c in cS:
        if c in cT:
            Xs_c = Xs[np.where(ys == c)[0],:]
            Xt_c = Xt[np.where(yt == c)[0],:]
            mmd += MMD(Xs_c, Xt_c,l)
    return mmd

def linear_MMD(Xs,Xt, l=None, type_flag='square'):

    ns, nt = Xs.shape[0],  Xt.shape[0]
    if l==None:
        l = median_heuristic(Xs, Xt, type_flag=type_flag)
        
    Ks = rbf(Xs, Xs, l)
    Kt = rbf(Xt, Xt, l)
    Kst = rbf(Xs, Xt, l)
    Kts = rbf(Xt, Xs, l)
    
    gi = Ks[::2, 1::2] + Kt[::2, 1::2] - Kst[::2, 1::2] - Kts[::2, 1::2]
    mmdapprox = np.sum(gi)

    return 2 * mmdapprox / Xt.shape[0]