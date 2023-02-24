import os
import multiprocessing as mp
import numpy as np
import pandas as pd

from ..src.data_handle import loadshape
from ..src.main_functions import hausdorff_quant

import argparse
from itertools import product

"""
all-dir is ./classes/train or ./classes/test
proto-csv contains shape NAMES, CLASSES, PATHS
resolution-level is one of s450, s900, s1800, s3600, s7200
"""

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--all-dir' , required=True) 
parser.add_argument('-p', '--proto-csv' , required=True)
parser.add_argument('-r', '--resolution-level' , required=True)
parser.add_argument('-q', '--haus-quants', nargs='*', type=float)
parser.add_argument('-w', '--haus-weights', nargs='*', type=float)
parser.add_argument('-n', '--num-workers', type=int)
parser.add_argument('-c', '--chunk-size', type=int)

args = parser.parse_args()
all_dir = args.all_dir
proto_csv_path = args.proto_csv
resol = args.resolution_level
haus_q = args.haus_quants
haus_w = args.haus_weights
n_workers = args.num_workers if args.num_workers else int(mp.cpu_count()//2)
chunk_size = args.chunk_size if args.chunk_size else 10

if haus_q is None and haus_w is None:
    haus_q = np.linspace(0,1,11)
    haus_w = np.ones_like(haus_q, dtype=float)
    haus_w[0] = 0.5
    haus_w[-1] = 0.5

elif haus_q and haus_w is None:
    haus_q = np.array(haus_q)
    haus_w = np.ones_like(haus_q)/len(haus_q)

elif haus_q and haus_w:
    haus_q = np.array(haus_q)
    haus_w = np.array(haus_w)

haus_q = np.round(haus_q,2)
haus_w = np.round(haus_w,2)

assert len(haus_q) == len(haus_w), \
    "Quantiles and weights should have the same length"

classes = [dir for dir in os.listdir(all_dir) if os.path.isdir(os.path.join(all_dir, dir))]

# S = dict() # shape arrays
shape_paths = dict()
shape_class = dict()

for cls in classes:
    class_dir = os.path.join(all_dir, cls)
    matfiles = [file for file in os.listdir(class_dir) if (file.endswith('.mat') and file.startswith('shapes_'))]
    for file in matfiles:
        shape = file.removesuffix('.mat').removeprefix('shapes_')
        shape_paths[shape] = os.path.join(class_dir, file)
        shape_class[shape] = cls

arrays = {shape:loadshape(sh_path, res=resol) for shape, sh_path in shape_paths.items()}

dfp = pd.read_csv(proto_csv_path)
proto_paths = {shape: shp for shape, shp in zip(dfp['shape'].values, dfp['path'].values)}
proto_class = {shape: cls for shape, cls in zip(dfp['shape'].values, dfp['class_'].values)}
proto_arrays = {shape:loadshape(sh_path, res=resol) for shape, sh_path in proto_paths.items()}

arrays.update(proto_arrays)
shape_class.update(proto_class)

pairs = tuple(product(shape_paths.keys(), proto_paths.keys()))
array_pairs = tuple((arrays[s1], arrays[s2]) for s1, s2 in pairs)

def dist(s1, s2):
    return hausdorff_quant(s1, s2, haus_q)

if __name__ == '__main__':
    mp.freeze_support()
    with mp.Pool(n_workers) as pool:
        results = pool.starmap(dist, array_pairs, chunksize=chunk_size)

    shapes_, protos_ = list(zip(*pairs))
    shape_classes_ = [shape_class[shape] for shape in shapes_]
    protos_classes_ = [shape_class[proto] for proto in protos_]
    quantiles = list(zip(*results))
    distance = [np.sum(q_val * haus_w) for q_val in results]

    data=dict(
        shape=shapes_,
        shape_class=shape_classes_,
        prototype=protos_,
        prot_class=protos_classes_,
        dist=distance
    )

    for q, q_val in zip(haus_q, quantiles):
        data[f'q{q}'] = q_val

    df = pd.DataFrame(data)
    proto_csv = os.path.basename(proto_csv_path)
    dir = os.path.basename(all_dir)
    df.to_csv(f'A2P-{dir}-{proto_csv}-{resol}.csv')

    





