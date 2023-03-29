import argparse
import time
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from scipy.stats import entropy
import numpy as np
import ot

from ac.src.main_functions import hausq_dist

parser = argparse.ArgumentParser()
parser.add_argument('-csv' , required=True) 
parser.add_argument('-a' , required=True, type=int) 
parser.add_argument('-b' , required=True, type=int) 
args = parser.parse_args()

csv_file = args.csv
a = args.a
b = args.b

df = pd.read_csv(csv_file)

b = min(b, len(df)-1)

df = df.loc[a:b]

for i in range(a,b+1):
    start = time.time()
    
    s1 = np.load(df.loc[i].path1)
    s2 = np.load(df.loc[i].path2)

    c1 = s1[:,:3]
    c2 = s2[:,:3]
    
    df.loc[i, 'ot'] = hausq_dist(c1, c2)
    end = time.time()
    df.loc[i, 'runtime'] = round(end-start,2)




df.to_csv(f"ot_shard_{a}_{b}.csv", index=None)