import argparse
import time
from main_functions import eval_d2d_dist


parser = argparse.ArgumentParser()
parser.add_argument('-d1', '--dir1' , required=True) 
parser.add_argument('-d2', '--dir2' , required=True) 
parser.add_argument('-r', '--resol' , required=True)
args = parser.parse_args()

df = eval_d2d_dist(args.dir1, args.dir2, args.resol)
df.to_csv(f'A2A-{args.dir1}-{args.dir2}-{args.resol}.csv')

