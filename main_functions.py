import numpy as np

def hausdorff_quant(x, y, q):
    assert isinstance(x, np.ndarray) and isinstance(y, np.ndarray), \
        "x and y have to be numpy arrays"
    assert x.shape[1] == y.shape[1] == 3, \
        "x and y have to be in 3d"
    D = x.reshape((-1,1,3)) - y.reshape((1,-1,3))
    D2 = np.sum(D**2,axis=2)
    Dx = D2.min(axis=0)
    Dy = D2.min(axis=1)
    Qx = np.quantile(Dx,q)
    Qy = np.quantile(Dy,q)
    Q = np.max((Qx, Qy), axis=0)
    return np.sqrt(Q)
