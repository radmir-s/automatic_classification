import numpy as np
import os
from collections import defaultdict
from itertools import product
import pandas as pd

from data_handle import loadshape

def hausdorff_quant(x, y, q=np.linspace(0,1,11)):
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

def hausq_dist(x, y, q = np.linspace(0,1,11)):
    w = np.ones_like(q, dtype=float)
    w[0] = 0.5
    w[-1] = 0.5

    quants = hausdorff_quant(x, y, q)

    return np.sum(quants * w)


def eval_d2d_dist(dir1, dir2, resol):

    # default summation parameters
    haus_q = np.linspace(0,1,11)
    haus_w = np.ones_like(haus_q, dtype=float)
    haus_w[0] = 0.5
    haus_w[-1] = 0.5

    matfiles1 = [file for file in os.listdir(dir1) if (file.endswith('.mat') and file.startswith('shapes_'))]
    # shape_paths1 = dict()
    shape_arrays1 = dict()
    for file in matfiles1:
        shape = file.removesuffix('.mat').removeprefix('shapes_')
        # shape_paths1[shape] = os.path.join(dir1, file)
        # shape_arrays1[shape] = loadshape(shape_paths1[shape], res=resol)
        shape_path = os.path.join(dir1, file)
        shape_arrays1[shape] = loadshape(shape_path, res=resol)

    matfiles2 = [file for file in os.listdir(dir2) if (file.endswith('.mat') and file.startswith('shapes_'))]
    # shape_paths2 = dict()
    shape_arrays2 = dict()
    for file in matfiles2:
        shape = file.removesuffix('.mat').removeprefix('shapes_')
        # shape_paths2[shape] = os.path.join(dir2, file)
        # shape_arrays2[shape] = loadshape(shape_paths1[shape], res=resol)
        shape_path = os.path.join(dir2, file)
        shape_arrays2[shape] = loadshape(shape_path, res=resol)

    out_data = defaultdict(lambda:list())

    for (s1, S1), (s2, S2) in product(shape_arrays1.items(), shape_arrays2.items()):
        out_data['shape1'].append(s1)
        out_data['shape2'].append(s2)
        out_data['dist'].append(hausq_dist(S1, S2))

    df = pd.DataFrame(out_data)

    return df


