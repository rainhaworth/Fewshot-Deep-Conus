# script to iterate over all decomposed files (see deep-conus/decompose.py) and extract max vals
# then store in csv
# the UH25 variable is not included in these files; see uh25max.csv for those max vals
import pickle
import numpy as np
import glob
import os
from tqdm import tqdm
import csv

indir = '/fs/nexus-scratch/rhaworth/deep-conus/future/'
filename = 'maxvals.csv'

print('shape and max for patch 0')
testfile = os.path.join(indir, 'patch_0.pickle')
with open(testfile, 'rb') as f:
    arr = pickle.load(f)
    print(np.shape(arr))
    print(np.max(arr, axis=(1,2)))
    print(len(np.max(arr, axis=(1,2))))

print('gathering filenames')
files = glob.glob(os.path.join(indir, 'patch_*.pickle'))

# open output file, create csv writer
with open(os.path.join(indir, filename), 'w') as csvf:
    writer = csv.writer(csvf)

    # write variable names
    writer.writerow(['tk_1km', 'tk_3km', 'tk_5km', 'tk_7km',
                     'ev_1km', 'ev_3km', 'ev_5km', 'ev_7km',
                     'eu_1km', 'eu_3km', 'eu_5km', 'eu_7km', 
                     'qv_1km', 'qv_3km', 'qv_5km', 'qv_7km',
                     'pr_1km', 'pr_3km', 'pr_5km', 'pr_7km',
                     'wmax', 'dbz', 'ctt'])
    
    # iterate over files
    for file in tqdm(files, 'computing max vals'):
        with open(file, 'rb') as f:
            sample = pickle.load(f)
            writer.writerow(np.max(sample, axis=(1,2)))