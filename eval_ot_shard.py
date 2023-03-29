import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from scipy.stats import entropy
import numpy as np
import ot

parser = argparse.ArgumentParser()
parser.add_argument('-csv' , required=True) 
parser.add_argument('-a' , required=True, type=int) 
parser.add_argument('-b' , required=True, type=int) 
args = parser.parse_args()

csv_file = args.csv
a = args.a
b = args.b

df = pd.read_csv(csv_file)
df = df.loc[a:b]

for i in range(a,b+1):
    start = time.time()
    
    s1 = np.load(df.loc[i].path1)
    s2 = np.load(df.loc[i].path2)

    c1 = s1[:,:3]
    c2 = s2[:,:3]

    d1 = np.ones(len(c1)) / len(c1)
    d2 = np.ones(len(c2)) / len(c2)
    M = np.sqrt( np.sum( np.square( c1.reshape(-1,1,3) - c2.reshape(1,-1,3) ), axis=2 ) )
    T = ot.emd(d1, d2, M)
    
    end = time.time()
    df.loc[i, 'runtime'] = round(end-start,2)
    df.loc[i, 'ot'] = np.sum(T*M)
    entt = entropy(T, axis=1)
    df.loc[i, 'entq90'] = np.quantile(entt, q=0.90)
    df.loc[i, 'entq99'] = np.quantile(entt, q=0.99)
    df.loc[i, 'np1'] = len(c1)
    df.loc[i, 'np2'] = len(c2)

df.to_csv(f"ot_shard_{a}_{b}.csv", index=None)