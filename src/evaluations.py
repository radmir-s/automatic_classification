import numpy as np
import os
import pandas as pd
import ot

from collections import defaultdict
from itertools import product

from .data_handle import loadshape
from .main_functions import hausq_dist, sinkhorn, emd, densify2


def s2s_regularity(s1, s2):

    cd1 = densify2(s1, cubenum=None)
    cd2 = densify2(s2, cubenum=None)

    c1 = cd1[:,:3]
    c2 = cd2[:,:3]

    # d1 = cd1[:,3].flatten()
    # d2 = cd2[:,3].flatten() 

    d1 = np.ones(len(cd1)) 
    d2 = np.ones(len(cd2)) / len(cd2) * len(cd1)

    M = np.sqrt( np.sum( np.square( c1.reshape(-1,1,3) - c2.reshape(1,-1,3) ), axis=2 ) )

    T = ot.emd(d1, d2, M)

    dpn = np.sum(T>0.,axis=1) # dest_point_num
    data = [np.mean(dpn), np.max(dpn), np.sum(dpn>2), np.sum(dpn>3)]

    ms = np.mean(np.sort(np.linalg.norm(c2.reshape(1,-1,3) - c2.reshape(-1,1,3),axis=2),axis=0)[1:4])

    centers =  T @ c2
    dist = np.linalg.norm(c2.reshape(1,-1,3)-centers.reshape(-1,1,3),axis=2) 
    radii = np.sum(T*dist,axis=1)

    data.extend((np.sum(radii>ms), np.sum(radii>2*ms), np.sum(radii>4*ms)))

    return data

def eval_d2d_regularity(dir1, dir2, resol='s7200'):

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
        dpnmean, dpnmax, dpn2, dpn3, rad1, rad2, rad3 = s2s_regularity(S1, S2)
        out_data['dpnmean'].append(dpnmean)
        out_data['dpnmax'].append(dpnmax)
        out_data['dpn2'].append(dpn2)
        out_data['dpn3'].append(dpn3)
        out_data['rad1'].append(rad1)
        out_data['rad2'].append(rad2)
        out_data['rad3'].append(rad3)

    df = pd.DataFrame(out_data)

    return df

def eval_d2d_dist(dir1, dir2, resol):

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

def eval_d2d_sinkhorn(dir1, dir2, resol, vox, reg=1e-2, numItermax=10_000):

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