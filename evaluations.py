import numpy as np
import os
import pandas as pd

from collections import defaultdict
from collections import defaultdict
from itertools import product

from data_handle import loadshape
from main_functions import hausq_dist, sinkhorn, emd

def eval_d2d_emd_2(dir1, dir2, resol, vox):
    pass

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


