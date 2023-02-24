import os
import numpy as np
from random import sample

from ..src.data_handle import loadshape
from ..src.main_functions import hausdorff_quant

import pandas as pd
import argparse
from collections import defaultdict
from k_means_constrained import KMeansConstrained
import json

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--train-dir' , required=True)
parser.add_argument('-c', '--class-num' , required=True)
parser.add_argument('-p', '--prot-perc',type=float, default=0.01)
parser.add_argument('-r', '--ran-init',type=float, default=0.05)
parser.add_argument('-q', '--haus-quants', nargs='*', type=float)
parser.add_argument('-w', '--haus-weights', nargs='*', type=float)

args = parser.parse_args()

# print(args)
# raise KeyError

ran_init = args.ran_init
cls_n = args.class_num
train_dir = args.train_dir
haus_q = args.haus_quants
haus_w = args.haus_weights
prot_perc = args.prot_perc

info = {}

###################### DISTANCES TO RANDOM SHAPES WITHIN CLASSES  ######################

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


class_dir = os.path.join(train_dir, cls_n)
matfiles = [file for file in os.listdir(class_dir) if file.endswith('.mat')]#[:10]
filepaths = {file.replace('.mat',''): os.path.join(class_dir, file) for file in matfiles}
info['filepaths'] = filepaths

S = {k: loadshape(v) for k, v in filepaths.items()}

ran_num = int(len(S)*ran_init)
random_shapes = sample(list(S.keys()), k=ran_num)
info['random_shapes'] = random_shapes

print(f'{len(S)}x{ran_num} distances are to be estimated')
print('Quantiles: ', *haus_q)
print('Weights: ', *np.round(haus_w,2))

info['quantiles'] = haus_q
info['weights'] = np.round(haus_w,2)


data = defaultdict(lambda: list())
more_data = defaultdict(lambda: list())

for s1 in S.keys():
    data['shapes'].append(s1)
    more_data['shapes'].append(s1)
    for s2 in random_shapes:
        q_vals = hausdorff_quant(S[s1], S[s2], haus_q)
        dist = np.sum(q_vals * haus_w)
        data[s2].append(dist)
        for q, v in zip(haus_q, q_vals):
            more_data[f'{s2}_{q}'].append(v)

df = pd.DataFrame(data)
dfm = pd.DataFrame(more_data)

df.sort_values('shapes').reset_index(drop=True)
dfm.sort_values('shapes').reset_index(drop=True)

distances_csv_path = f'distances_random_{cls_n}_{ran_num}.csv'
distances_csv_path_more = f'distances_random_{cls_n}_{ran_num}_more.csv'

df.to_csv(distances_csv_path)
dfm.to_csv(distances_csv_path_more)
info['distances_csv_path'] = distances_csv_path
info['distances_csv_path_more'] = distances_csv_path_more


###################### CLUSTERING WITHIN CLASSES (PROTOTYPES)  ######################

X = df.drop('shapes',axis=1).to_numpy()
n_clust = int(X.shape[0] * prot_perc)
n_feat = X.shape[1]

kmc = KMeansConstrained(
                        n_clusters=n_clust, 
                        init='random', 
                        size_min=30, 
                        size_max=300, 
                        n_init=30, 
                        n_jobs=-1
                        )
kmc.fit(X)

centers = kmc.cluster_centers_
D = X.reshape((1,-1,n_feat))-centers.reshape((-1,1,n_feat))
center_shapes = np.argmin(np.sum(D**2,axis=2),axis=1)
center_shapes.sort()
dfc = df.iloc[center_shapes].shapes

prototypes_csv_path = f'prototypes_{cls_n}_{n_clust}.csv'

dfc.to_csv(prototypes_csv_path)
info['prototypes'] = dfc.to_list()
info['prototypes_csv_path'] = prototypes_csv_path

with open('info_{cls_n}.json', 'w') as fp:
    json.dump(info, fp)
