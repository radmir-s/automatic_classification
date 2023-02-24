import argparse
import time
from main_functions import eval_d2d_sinkhorn


parser = argparse.ArgumentParser()
parser.add_argument('-d1', '--dir1' , required=True) 
parser.add_argument('-d2', '--dir2' , required=True) 
parser.add_argument('-r', '--resol' , required=True)
parser.add_argument('-v', '--vox' , required=True, type=float)
parser.add_argument('-l', '--reg' , required=True, type=float)
parser.add_argument('-i', '--max-iter' , required=True, type=int)
args = parser.parse_args()

start=time.time()
df = eval_d2d_sinkhorn(args.dir1, args.dir2, args.resol, args.vox, args.reg, args.max_iter)
end=time.time()
duration = int(end - start)

df.to_csv(f'd2d-sinkhorn-{duration}-{args.vox}-{args.reg}.csv')


"""
command use template:

python eval_d2d_sinkhorn.py -d1 $dir1 -d2 $dir2 -r s7200  -v 0.05 -l 0.01 -i 5000 
"""