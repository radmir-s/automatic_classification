import os
from data_handle import loadshape
import numpy as np
from random import sample
from main_functions import hausdorff_quant
import pandas as pd
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()

parser.add_argument('-d', '--class-dir' , required=True)
parser.add_argument('-i', '--ran-init',type=float, default=0.05)
parser.add_argument('-q', '--haus-quants', nargs='*', type=float, default=[0.35, 0.55 ,0.75, 0.95])
parser.add_argument('-w', '--haus-weights', nargs='*')

args = parser.parse_args()

# print(args)
# raise KeyError

ran_init = args.ran_init
class_dir = args.class_dir
haus_q = args.haus_quants

if args.haus_weights is None or args.haus_weights[0] == 'auto':
    haus_q = sorted(haus_q)
    haus_w = np.diff(haus_q+[1.])
elif args.haus_weights[0] == 'equal':
    haus_w = np.ones_like(haus_q)/len(haus_q)
else:
    haus_w = np.array(args.haus_weights)

assert len(haus_q) == len(haus_w), \
    "Quantiles and weights should have same length"

matfiles = [file for file in os.listdir(class_dir) if file.endswith('.mat')][:30]
filepaths = {file.replace('.mat',''): os.path.join(class_dir, file) for file in matfiles}

S = {k: loadshape(v) for k, v in filepaths.items()}

ran_num = int(len(S)*ran_init)
random_sample = sample(list(S.keys()), k=ran_num)

print(f'{len(S)}x{ran_num} distances are to be estimated')
print('Quantles: ', *haus_q)
print('Weights: ', *np.round(haus_w,2))

# quantles = lambda x, y: hausdorff_quant(x,y, haus_q)
# comb_quant = lambda q: np.sum(q * haus_w)

data = defaultdict(lambda: list())
more_data = defaultdict(lambda: list())

for s1 in S.keys():
    data['shapes'].append(s1)
    more_data['shapes'].append(s1)
    for s2 in random_sample:
        q_vals = hausdorff_quant(S[s1], S[s2], haus_q)
        dist = np.sum(q_vals * haus_w)
        data[s2].append(dist)
        for q, v in zip(haus_q, q_vals):
            more_data[f'{s2}_{q}'].append(v)

df = pd.DataFrame(data)
dfm = pd.DataFrame(more_data)

cls = '_'.join(class_dir.split('/')[-2:])
df.to_csv(f'distances_random_{cls}_{ran_init}.csv')
dfm.to_csv(f'distances_random_{cls}_{ran_init}_more.csv')