"""
Simple alignment method for DA under class imbalance: 
"On statistic alignment for domain adaptation in structural health monitoring, J Poole et al. (2020)"

@author: Jack Poole

"""
import numpy as np 

class NCA:
    def __init__(self):
        pass
    
    def fit(self, Xs,Xt,ys,yt):
        #inputs:
        #Xs- source data (ns,d)
        #Xt -target data (nt,d)
        #ys - labels; data aligned via class 0 (ns,1) (or labels in common between source and target)
        #yt - labels; data aligned via class 0 (nt,1)
        #Note: if more than one class is used the labels should be balanced
        
        #1) normalise the source domain
        self.mu_s=np.mean(Xs,axis=0) 
        self.std_s=np.std(Xs,axis=0)
        Xs_=(Xs.copy() - self.mu_s) / self.std_s 
        
        #2) estiamte mu and std of the normal condition (class 0)
        Xs_n=Xs_[np.where(ys == 0)[0],:]
        self.mu_sn =np.mean(Xs_n,axis=0) 
        self.std_sn =np.std(Xs_n,axis=0)
            
        Xt_n=Xt[np.where(yt == 0)[0],:]
        self.mu_tn=np.mean(Xt_n,axis=0)
        self.std_tn=np.std(Xt_n,axis=0)
        
    def transform_s(self, X):
        
        #standardise 
        return (X - self.mu_s) / self.std_s
        
    def transform_t(self, X):
        
        # align the target
        return (X - self.mu_tn) * (self.std_sn / self.std_tn) + self.mu_sn
        
       