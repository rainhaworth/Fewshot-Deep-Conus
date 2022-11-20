import os
import pickle
import numpy as np
import xarray as xr
import argparse
from tqdm import tqdm

# Decompose netcdf into individual .pickle files
# The resulting files should be slightly smaller than the input files
def decompose_nc(datasrc='./data/', filestr='spatial_storm_data_part', outdir='./data/', start=1):
    print("Decomposing files into (serialized) ndarrays")

    # Track patch index across files
    filenum = start
    idx = (1-start) * 90000 # NOTE: This assumes every file up to start has exactly 90k patches

    # Populate array with max uh25 values
    uh25_max_arr = []

    # Traverse all files
    while os.path.isfile(datasrc + filestr + str(filenum) + '.nc'):
        ds = xr.open_dataset(datasrc + filestr + str(filenum) + '.nc')
        
        # Calculate and store max UH values for the entire file
        uh25mx = ds['uh25'].max(['y','x'],skipna=True).values
        uh25_max_arr.extend(uh25mx)

        ds = ds.drop_vars('uh25')
        data_array_list = [ds[key] for key in list(ds)]

        # NOTE: it wouldn't be hard to achieve performance gains w/ parallelism here
        # Currently this takes around 15 minutes per file (90k patches)
        for j in tqdm(range(len(uh25mx)), desc=('File ' + str(filenum))):
            # if it hasn't been created yet, create a new 23x32x32 tensor, serialize, and write to disk
            if not os.path.isfile(outdir + 'patch_' + str(idx) + '.pickle'):
                _elem = np.stack([data_array_list[i][j].values for i in range(len(data_array_list))])

                with open(outdir + 'patch_' + str(idx) + '.pickle','wb') as f:
                    pickle.dump(_elem, f)
            
            idx += 1

        filenum += 1
    
    # After all patches have been serialized, write uh_25_max_arr as a .csv
    with open(outdir + 'uh25max.csv','wb') as f:
        np.savetxt(f, uh25_max_arr, delimiter=',')

# Parse command line args
parser = argparse.ArgumentParser(description='Split and pickle dataset.')
parser.add_argument('--datasrc',    type=str,   default='./data/',                  help='datset source directory')
parser.add_argument('--filestr',    type=str,   default='spatial_storm_data_part',  help='filename substring (expects $(filestr)X.nc, X = 1..n)')
parser.add_argument('--outdir',     type=str,   default='./data/',                  help='directory to store output files')
parser.add_argument('--start',      type=int,   default=1,                          help='file number to start decomposing')

args = parser.parse_args()

# Run function
decompose_nc(datasrc=args.datasrc, filestr=args.filestr, outdir=args.outdir, start=args.start)