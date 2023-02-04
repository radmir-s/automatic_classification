import argparse
import time
from main_functions import eval_d2d_dist


parser = argparse.ArgumentParser()
parser.add_argument('-d1', '--dir1' , required=True) 
parser.add_argument('-d2', '--dir2' , required=True) 
parser.add_argument('-r', '--resol' , required=True)
args = parser.parse_args()

start=time.time()
df = eval_d2d_dist(args.dir1, args.dir2, args.resol)
end=time.time()
duration = int(end - start)

df.to_csv(f'A2A-{duration}.csv')


"""
command use template:

python eval_d2d_dist.py -d1 dir1 -d2 dir2 -r s900
"""