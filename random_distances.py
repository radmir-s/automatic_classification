import os
from data_handle import loadshape
import numpy as np
from random import sample
from main_functions import hausdorff_quant
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--class-dir' , required=True)
parser.add_argument('-i', '--ran-init',type=float, default=0.05)
parser.add_argument('-q', '--haus-quants', nargs='*', type=float)
parser.add_argument('-w', '--haus-weights', nargs='*')

args = parser.parse_args()

# ran_init = 0.05
# class_dir = './classes/train/0/'
# haus_q = [0.35, 0.55 ,0.75, 0.95]
# haus_w = [1., 1., 1., 1.]

ran_init = args.ran_init
class_dir = args.class_dir
haus_q = args.haus_quants if args.haus_quants else [0.35, 0.55 ,0.75, 0.95]
haus_w = args.haus_weights if args.haus_weights else [1., 1., 1., 1.]

assert len(haus_q) == len(haus_w), \
    "Quantiles and weights should have same length"

matfiles = [file for file in os.listdir(class_dir) if file.endswith('.mat')]
filepaths = {file.replace('.mat',''): os.path.join(class_dir, file) for file in matfiles}

S = {k: loadshape(v) for k, v in filepaths.items()}

ran_num = int(len(S)*ran_init)
random_sample = sample(list(S.keys()), k=ran_num)

dist = lambda x, y: np.sum(hausdorff_quant(x,y, haus_q) * np.array(haus_w))

data = {s:list() for s in random_sample}
data['shapes'] = list()

for s1 in S.keys():
    data['shapes'].append(s1)
    for s2 in random_sample:
        d = dist(S[s1], S[s2])
        data[s2].append(d)


df = pd.DataFrame(data)
cls = '_'.join(class_dir.split('/')[-2:])
df.to_csv(f'distances_random_{cls}_{ran_init}.csv')