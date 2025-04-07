"""
Implementation of transfer component analysis (TCA) introduced in:
Domain adaptation via transfer component analysis, SJ Pan et al. (2010)

@author: Jack Poole

"""

import numpy as np
from scipy.spatial.distance import cdist
import scipy.linalg
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import pairwise_distances
from sklearn.metrics import f1_score

from TL_models.dist_measures import median_heuristic
from TL_models.standard_checks import check_inputs



class TCA():
    """Transfer Component anaylsis 
    Inputs: classifier - any sklearn supervised classifier
            params     - dictionary of hyperparamters:
                        'mu'     - regularisation hyperparameter
                        'k'      - number of transfer components to return (k<d)
                        'kernel' - linear, poly, rbf kernel
                        Optional:
                        'gamma','b', 'M' - for poly kernel: (gamma*<X,Y>+b)^M
                        'ls'             - bandwidth for rbf, if unspecified 
                                           it uses the median heuristic
                       """
    def __init__(self, classifier=KNeighborsClassifier(n_neighbors=1),
                 params={'lmda': 0.1, 'k': 2, 'kernel': 'rbf', 'ls':None}, 
                 type_flag='abs'):

        self.params = params
        self.W = None
        self.X = None
        self.classifier = classifier
        try:
            self.length_scale = params['ls']
        except:
            self.length_scale = None
            self.type_flag = type_flag
        
    def get_kernel(self, X, Y=None, length_scale = None):
        """Calculate kernel matrix between X and Y."""
        kernel = self.params['kernel']
        # If Y is None calculate the kernel with itself
        if Y is None:
            Y = X

        # return kernel matrix
        if kernel == 'linear':
            return np.dot(X, Y.T)
        elif kernel == 'cosine':
            return np.dot(X, Y.T)/ (np.linalg.norm(X)*np.linalg.norm(Y))
        elif kernel == 'poly':
            (self.params['gamma']*np.dot(X, Y.T) +
             self.params['b'])**self.params['M']
        elif kernel == 'rbf':
            if length_scale == None and self.length_scale == None:
                D = pairwise_distances(X, Y)
                length_scale = np.median(D)
                self.length_scale = median_heuristic(X, Y, self.type)
            else: 
                length_scale = self.length_scale
                
            return np.exp(-np.square(cdist(X, Y, metric='euclidean')) / (2*(length_scale)**2))
        else:
            raise ValueError('Invalid kernel')
            
    def get_L(self, n_s, n_t):
        """Creates the MMD matrix"""
        n_s_ones = 1.0/n_s*np.ones((n_s, 1))
        n_t_ones = -1.0/n_t*np.ones((n_t, 1))
        n_stack = np.vstack((n_s_ones, n_t_ones))
        L = np.dot(n_stack, n_stack.T)
        return L

    def get_H(self, n):
        """Creates the centering matrix"""
        return np.eye(n) - 1./n * np.ones((n, n))

    def fit(self, X_s, X_t):
        """Learns the transformation W

        Inputs: X_s - matrix of source samples (ns x d)
                X_t - matrix of target samples (nt x d)"""

        lmda, k = self.params['lmda'], self.params['k']
        n_s, d = X_s.shape
        n_t, d = X_t.shape
        self.X = np.vstack((X_s, X_t))
        n = n_s + n_t

        if k > d:
            raise ValueError('Requested too many dimensions!')

        #get mmd, centering and kernel matrix
        L = self.get_L(n_s, n_t)
        H = self.get_H(n)
        if self.params['MK']:
            K = self.get_multi_kernel(X_s, X_t, n_s, n_t)
        else:
            K = self.get_kernel(self.X, Y=None)

        if np.linalg.det(K) == 0:
            #print('Reguralising K...')
            count = 0
            # regularise K, what is the implication of this
            while np.linalg.det(K) == 0 and count < 10:
                K += np.eye(K.shape[0])*1*10**-6
                count += 1

        mini = np.dot(np.dot(K.T, L), K)+lmda*np.eye(n)  # minimise MMD distance
        st = np.dot(np.dot(K.T, H), K)  # subject to variance =1

        # Solves the pairwise eigenvalue problem (assumes det(A-lambda*B)=0), more robust because no inversion
        eigval, eigvec = scipy.linalg.eig(mini, st)
        index = np.argsort(np.absolute(eigval)) 
        # plt.plot(eigval)
        # Full rank nxn transformation matrix
        self.W = np.real(eigvec[:, index][:, :k])

        Z = np.dot(K, self.W)
        Z_s = Z[:n_s, :]
        Z_t = Z[n_s:, :]
        return Z_s, Z_t

    def transform(self, X_test):

        K = self.get_kernel(X_test, self.X)
        Z = np.dot(K, self.W)

        return Z

    def train(self, X_s, y_s, classifier=KNeighborsClassifier()):
        '''Train a classifier using fitted data'''
        Z_s = self.transform(X_s)
        
        if np.unique(y_s).shape[0] > 2:
            # Allows any classifier to be used
            self.binary = False
            self.classifier = OneVsRestClassifier(
                classifier).fit(Z_s, np.ravel(y_s))
        else:
            self.classifier = classifier.fit(Z_s, np.ravel(y_s))

    def predict(self, X_test, y_t=None):
        '''Predict classes in the target domain'''
        Z = self.transform(X_test)
        pred = self.classifier.predict(Z)
        if y_t is not None and not self.binary:
            acc = self.classifier.score(Z, np.ravel(y_t))
            f1 = f1_score(np.ravel(y_t), pred, average='macro')
            return pred, acc, f1
        elif y_t is not None:
            acc = self.classifier.score(Z, np.ravel(y_t))
            f1 = f1_score(np.ravel(y_t), pred)
            return pred, acc, f1
        else:
            return pred


