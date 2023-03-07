import argparse
import time
from src.evaluations import eval_d2d_regularity

parser = argparse.ArgumentParser()
parser.add_argument('-d1', '--dir1' , required=True) 
parser.add_argument('-d2', '--dir2' , required=True) 
args = parser.parse_args()

start=time.time()
df = eval_d2d_regularity(args.dir1, args.dir2)
end=time.time()
duration = int(end - start)

df.to_csv(f'd2d-regularity-{duration}.csv')


"""
command use template:

python eval_d2d_regularity.py -d1 $dir1 -d2 $dir2
"""