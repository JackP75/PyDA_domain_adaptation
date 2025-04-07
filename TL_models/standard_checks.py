
import numpy as np

def check_inputs(Xs, ys, Xt, yt=None):
    """
    Checks the input data for DA methods
    """
    
    if not isinstance(Xs, np.ndarray):
        raise ValueError("Xs should be a numpy array.")
    if not isinstance(Xt, np.ndarray):
        raise ValueError("Xt should be a numpy array.")
    if not isinstance(ys, np.ndarray):
        raise ValueError("ys should be a numpy array.")
    if yt is not None and not isinstance(yt, np.ndarray):
        raise ValueError("yt should be a numpy array when provided.")
    
    if Xs.shape[1] != Xt.shape[1]:
        raise ValueError("Xs and Xt should have the same number of features.")
    
    if ys.shape[0] != Xs.shape[0]:
        raise ValueError("The number of labels in ys should match the number of samples in Xs.")
    if yt is not None and yt.shape[0] != Xt.shape[0]:
        raise ValueError("The number of labels in yt should match the number of samples in Xt.")
    
    if np.any(np.isnan(Xs)) or np.any(np.isnan(ys)) or np.any(np.isnan(Xt)) or (yt is not None and np.any(np.isnan(yt))):
        raise ValueError("Inputs contain NaN values. Please handle missing values.")
    
    if Xs.shape[0] == 0 or Xt.shape[0] == 0:
        raise ValueError("Xs and Xt should contain at least one sample.")
    
