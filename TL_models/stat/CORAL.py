"""
Implementation of correlation alignment (CORAL) presented in:
Correlation alignment for unsupervised domain adaptation, B Sun et al. (2017)

@author: Jack Poole
"""

import numpy as np

class CORAL:
    "Before applying CORAL the data should be standardised to align the mean and std"
    def fit(self, Xs, Xt,ys=None,yt=None):
        "Finds the transformation matrix for CORAL"
        d=Xs.shape[1]
        Cs=np.cov(Xs.T)+np.eye(d,d)
        Ct=np.cov(Xt.T)+np.eye(d,d)
        Cs_rt=np.linalg.cholesky(Cs)
        Ct_rt=np.linalg.cholesky(Ct)
        self.A=np.linalg.inv(Cs_rt).dot(Ct_rt)
        return self.A
    
    def fit_normal(self,Xs,Xt,ys,yt):
        """Finds the transformation matrix for NCORAL, a modification to handle class imbalance from:
        On statistic alignment for domain adaptation in structural health monitoring, J Poole et al. (2020)"""
        d=Xs.shape[1]
        Xs_n=Xs[np.where(ys == 0)[0],:]
        Xt_n=Xt[np.where(yt == 0)[0],:]
        
        Cs=np.cov(Xs_n.T)+np.eye(d,d)
        Ct=np.cov(Xt_n.T)+np.eye(d,d)
        Cs_rt=np.linalg.cholesky(Cs)
        Ct_rt=np.linalg.cholesky(Ct)
        self.A=np.linalg.inv(Cs_rt).dot(Ct_rt)


    def transform(self,X):
        return X.dot(self.A)
    
    
    
    
        
        
 