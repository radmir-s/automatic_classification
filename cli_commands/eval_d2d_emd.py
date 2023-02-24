import argparse
import time
from main_functions import eval_d2d_emd


parser = argparse.ArgumentParser()
parser.add_argument('-d1', '--dir1' , required=True) 
parser.add_argument('-d2', '--dir2' , required=True) 
parser.add_argument('-r', '--resol' , required=True)
parser.add_argument('-v', '--vox' , required=True, type=float)
args = parser.parse_args()

start=time.time()
df = eval_d2d_emd(args.dir1, args.dir2, args.resol, args.vox)
end=time.time()
duration = int(end - start)

df.to_csv(f'd2d-emd-{duration}-{args.vox}.csv')


"""
command use template:

python eval_d2d_emd.py -d1 $dir1 -d2 $dir2 -r s7200  -v 0.05
"""