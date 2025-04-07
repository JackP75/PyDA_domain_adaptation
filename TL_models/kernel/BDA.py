"""
Implementation of balanced distribution adaptation (BDA) introduced in:
Balanced Distribution Adaptation for Transfer Learning, Jindong Wang et al. (2017)

@author: Jack Poole

"""

import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
import scipy.linalg

from TL_models.standard_checks import check_inputs
from .TCA import TCA

class BDA(TCA):
    def __init__(self,params={'classifier':KNeighborsClassifier(),
                              'kernel':'rbf',
                              'lmda':0.1,
                              'k':2,
                              'max_iter':5,
                              'mu':0.5,
                              'WBDA':False,
                                'ls':None}):
        
        super().__init__(params=params)   
        self.y_dist = None
        
    
    def get_M(self, n_s, n_t, s_prob=None, t_prob=None, class_mask=1):
        
        if type(class_mask)!=int:
            nsc = np.where(class_mask[:n_s] != 0)[0].shape[0]
            ntc = np.where(class_mask[n_s:] != 0)[0].shape[0]  
        else:
            nsc = n_s
            ntc = n_t
            s_prob = 1
            t_prob = 1
            
        M = np.zeros((n_s + n_t, n_s + n_t))
        M[:nsc, :nsc] = s_prob / nsc**2
        M[:nsc, nsc:] = (s_prob*t_prob)**(1/2) / (nsc * ntc)
        M[nsc:, :nsc] = (s_prob*t_prob)**(1/2) / (nsc * ntc)
        M[:nsc, :nsc] = t_prob / ntc**2
        
        return M

    def get_M_sum(self, M_dict):
        
        M = np.zeros(M_dict['M_0'].shape)
        for label, Mn in M_dict.items(): 
            M += Mn
        return M

    def fit(self, X_s, X_t, y_s):

        check_inputs(X_s, y_s, X_t, y_t)

        
        n_s = X_s.shape[0]       #source examples
        n_t = X_t.shape[0]       #target examples
        n = n_s + n_t
        I = np.eye(n)            #Identity n x n
        H = self.get_H(n)     #Centering matrix

        self.X = np.vstack((X_s,X_t))
        k = self.params['k']
        classes = np.unique(y_s) #array of class labels
        C = classes.shape[0]
        if classes.shape[0] > 2:
            self.binary=False    
        
        #class conditional probabilities
        if self.params['WBDA']:
            s_prob = np.zeros((classes.shape))
            for c in classes:
                s_prob[c] = y_s[np.where(c == y_s)].shape[0] / n_s
        else:
            s_prob = np.ones((C,1))
        
        if self.params['kernel'] is not None:
            X=self.get_kernel(self.X)
        
        ###initialise MMD matrix for marginal dists
        M_dict={}
        M_dict['M_0']=(1-self.params['mu'])*self.get_M(n_s,n_t) 
      
        ###get pseudo labels
        self.classifier = OneVsRestClassifier(self.classifier).fit(X_s, 
                                                                    y_s.reshape((n_s,)))
        y_t=self.classifier.predict(X_t)
        y=np.vstack((y_s.reshape(-1,1),y_t.reshape(-1,1)))
        
        #target conditional probs
        if self.params['WBDA']:
            t_prob = np.zeros((classes.shape))
            for c in classes:
                t_prob[c]=y_t[np.where(c==y_t)].shape[0]/n_t     
        else:
            t_prob = np.ones((classes.shape))
        
        ####initialise MMD matrix for conditional dists    
        for c in classes.astype(int): 
            index=np.zeros((n,1))
            index[np.argwhere(y == c)] = 1
            if c!=0:
                M_dict['M_'+str(c+1)] = self.params['mu'] * self.get_M(n_s,
                                                                       n_t, 
                                                                       s_prob[c], 
                                                                       t_prob[c], 
                                                                       index) 
        
            
        ### main loop 
        count = 0
        while count < self.params['max_iter']:
        
            count += 1
            
            ### adapt
            M = self.get_M_sum(M_dict)
            obj = np.dot(np.dot(X.T, M), X) + self.params['lmda'] * I
            st = np.dot(np.dot(X.T, H), X)

            eigval, eigvec = scipy.linalg.eig(obj,st)
            index = np.argsort(eigval)               
            self.W = np.real(eigvec[:,index][:,:k])         #Full rank nxn transformation matrix                           
            
            Z = np.dot(X,self.W)
            Z_s = Z[:n_s,:]
            Z_t = Z[n_s:,:]
       
            ### recalculate pseudo labels
            self.classifier = OneVsRestClassifier(self.classifier).fit(Z_s, 
                                                                       y_s.reshape((n_s,)))
            y_t = self.classifier.predict(Z_t)
            
            y = np.vstack((y_s.reshape(-1,1), y_t.reshape(-1,1)))
            
            
            if self.params['WBDA']:
                t_prob = np.ones((C,1))
                for c in classes:
                    t_prob[c] = y_t[np.where(c == y_t)].shape[0] / n_t     
            else:
                t_prob = np.ones((C,1))
                    
            for c in classes: 
                index = np.zeros((n,1))
                index[np.argwhere(y == c)] = 1
                M_dict['M_' + str(c+1)] = self.params['mu'] * self.get_M(n_s,
                                                                         n_t,
                                                                         s_prob[c],
                                                                         t_prob[c],
                                                                         index)

        return Z_s,Z_t
    

