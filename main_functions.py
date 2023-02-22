import numpy as np
import os
from collections import defaultdict
from itertools import product
import pandas as pd
import time
import ot

from data_handle import loadshape


def eval_d2d_sinkhorn(dir1, dir2, resol, vox, reg=1e-2, numItermax=10_000):

    # default summation parameters
    haus_q = np.linspace(0,1,11)
    haus_w = np.ones_like(haus_q, dtype=float)
    haus_w[0] = 0.5
    haus_w[-1] = 0.5

    matfiles1 = [file for file in os.listdir(dir1) if (file.endswith('.mat') and file.startswith('shapes_'))]
    shape_arrays1 = dict()
    for file in matfiles1:
        shape = file.removesuffix('.mat').removeprefix('shapes_')
        shape_path = os.path.join(dir1, file)
        shape_arrays1[shape] = loadshape(shape_path, res=resol)

    matfiles2 = [file for file in os.listdir(dir2) if (file.endswith('.mat') and file.startswith('shapes_'))]
    shape_arrays2 = dict()
    for file in matfiles2:
        shape = file.removesuffix('.mat').removeprefix('shapes_')
        shape_path = os.path.join(dir2, file)
        shape_arrays2[shape] = loadshape(shape_path, res=resol)

    out_data = defaultdict(lambda:list())

    for (s1, S1), (s2, S2) in product(shape_arrays1.items(), shape_arrays2.items()):
        out_data['shape1'].append(s1)
        out_data['shape2'].append(s2)
        cost, runtime, itern, (np1, np2) = sinkhorn(S1, S2, vox=vox, reg=reg, numItermax=numItermax)
        out_data['cost'].append(cost)
        out_data['runtime'].append(runtime)
        out_data['itern'].append(itern)
        out_data['np1'].append(np1)
        out_data['np2'].append(np2)

    df = pd.DataFrame(out_data)

    return df

def eval_d2d_emd(dir1, dir2, resol, vox):

    # default summation parameters
    haus_q = np.linspace(0,1,11)
    haus_w = np.ones_like(haus_q, dtype=float)
    haus_w[0] = 0.5
    haus_w[-1] = 0.5

    matfiles1 = [file for file in os.listdir(dir1) if (file.endswith('.mat') and file.startswith('shapes_'))]
    shape_arrays1 = dict()
    for file in matfiles1:
        shape = file.removesuffix('.mat').removeprefix('shapes_')
        shape_path = os.path.join(dir1, file)
        shape_arrays1[shape] = loadshape(shape_path, res=resol)

    matfiles2 = [file for file in os.listdir(dir2) if (file.endswith('.mat') and file.startswith('shapes_'))]
    shape_arrays2 = dict()
    for file in matfiles2:
        shape = file.removesuffix('.mat').removeprefix('shapes_')
        shape_path = os.path.join(dir2, file)
        shape_arrays2[shape] = loadshape(shape_path, res=resol)

    out_data = defaultdict(lambda:list())

    for (s1, S1), (s2, S2) in product(shape_arrays1.items(), shape_arrays2.items()):
        out_data['shape1'].append(s1)
        out_data['shape2'].append(s2)
        cost, runtime, itern, (np1, np2) = emd(S1, S2, vox=vox)
        out_data['cost'].append(cost)
        out_data['runtime'].append(runtime)
        out_data['itern'].append(itern)
        out_data['np1'].append(np1)
        out_data['np2'].append(np2)

    df = pd.DataFrame(out_data)

    return df


def emd(s1, s2, vox=5e-2):
    start = time.time()
    cd1 = densify(s1, vox=vox)
    cd2 = densify(s2, vox=vox)

    c1 = cd1[:, :3]
    d1 = cd1[:, 3:].flatten()

    c2 = cd2[:, :3]
    d2 = cd2[:, 3:].flatten()   
                
    M = np.sqrt( np.sum( np.square( c1.reshape(-1,1,3) - c2.reshape(1,-1,3) ), axis=2 ) )

    T, log = ot.emd(d1, d2, M, log=True)

    end = time.time()
    runtime = end-start

    return np.sum(T*M), runtime, None, T.shape



def sinkhorn(s1, s2, vox=5e-2, reg=5e-3, numItermax=10_000):
    start = time.time()
    cd1 = densify(s1, vox=vox)
    cd2 = densify(s2, vox=vox)

    c1 = cd1[:, :3]
    d1 = cd1[:, 3:].flatten()

    c2 = cd2[:, :3]
    d2 = cd2[:, 3:].flatten()   
                
    M = np.sqrt( np.sum( np.square( c1.reshape(-1,1,3) - c2.reshape(1,-1,3) ), axis=2 ) )

    T, log = ot.sinkhorn(d1, d2, M, reg=1e-2, method='sinkhorn', log=True, numItermax=numItermax)


    end = time.time()
    runtime = end-start
    
    return np.sum(T*M), runtime, log['niter'], T.shape



def densify(s, **kwargs):
    
    x1, y1, z1 = s.min(axis=0) - 1e-2
    x2, y2, z2 = s.max(axis=0) + 1e-2
    
    if 'vox' in kwargs:
        vox = kwargs['vox']
        xx = np.arange(x1, x2, vox)
        yy = np.arange(y1, y2, vox)
        zz = np.arange(z1, z2, vox)

        dimx = len(xx) - 1
        dimy = len(yy) - 1
        dimz = len(zz) - 1
    elif 'dim' in kwargs:
        dimx, dimy, dimz = kwargs['dim']
        xx = np.linspace(x1,x2,dimx+1)
        yy = np.linspace(y1,y2,dimy+1)
        zz = np.linspace(z1,z2,dimz+1)
    else:
        raise KeyError("Either 'vox' or 'dim' parameters should be defined")

    sx = s[:,0].reshape(1,-1)
    sy = s[:,1].reshape(1,-1)
    sz = s[:,2].reshape(1,-1)

    indx = (xx[:-1].reshape(-1,1) < sx) & (sx < xx[1:].reshape(-1,1))
    indy = (yy[:-1].reshape(-1,1) < sy) & (sy < yy[1:].reshape(-1,1))
    indz = (zz[:-1].reshape(-1,1) < sz) & (sz < zz[1:].reshape(-1,1))

    voxel = np.sum(indx.reshape(dimx,1,1,-1) * indy.reshape(1,dimy,1,-1) * indz.reshape(1,1,dimz,-1), axis=-1 )
    voxel = voxel/voxel.sum()

    x = (xx[:-1] + xx[1:])/2
    y = (yy[:-1] + yy[1:])/2
    z = (zz[:-1] + zz[1:])/2

    histocloud = [[x[indx], y[indy], z[indz], voxel[indx, indy, indz]] for indx, indy, indz in np.argwhere(voxel)]
    histocloud = np.array(histocloud)

    return histocloud


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

def hausq_dist(x, y, q = np.linspace(0,1,11)):
    w = np.ones_like(q, dtype=float)
    w[0] = 0.5
    w[-1] = 0.5
    quants = hausdorff_quant(x, y, q)

    return np.sum(quants * w)

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
