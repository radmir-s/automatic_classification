import os
import multiprocessing as mp
import numpy as np
import pandas as pd
from data_handle import loadshape
from main_functions import hausdorff_quant
import argparse
from collections import defaultdict
from itertools import product
import json
import time

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--train-dir' , required=True)
parser.add_argument('-q', '--haus-quants', nargs='*', type=float)
parser.add_argument('-w', '--haus-weights', nargs='*', type=float)
parser.add_argument('-n', '--num-workers', type=int)
parser.add_argument('-c', '--chunk-size', type=int)

args = parser.parse_args()
# print(args)
train_dir = args.train_dir
haus_q = args.haus_quants
haus_w = args.haus_weights
n_workers = args.num_workers if args.num_workers else (mp.cpu_count()-2)
chunk_size = args.chunk_size if args.chunk_size else 5

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
    "Quantiles and weights should have same length"

classes = os.listdir('./classes/train/')

# S = dict() # shape arrays
shape_class = dict()
shape_path = dict()

for cls in classes:
    class_dir = os.path.join(train_dir, cls)
    matfiles = [file for file in os.listdir(class_dir) if file.endswith('.mat')]
    for file in matfiles:
        shape = file.replace('.mat','')
        shape_path[shape] = os.path.join(class_dir, file)
        shape_class[shape] = cls

protos = list()
for cls in classes:
    with open(f'./info_{cls}.json', 'r') as fp:
        info = json.load(fp)
    prototypes_csv_path = info['prototypes_csv_path']
    df = pd.read_csv(prototypes_csv_path)
    protos.extend(df.shapes.to_list())

pairs = tuple(product(shape_path.keys(), protos))#[:10]
arrays = {shape:loadshape(sh_path) for shape, sh_path in shape_path.items()}
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
    df.to_csv('A2P.csv')

    





