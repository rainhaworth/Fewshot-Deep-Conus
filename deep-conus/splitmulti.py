# Split function with validation for multiple .nc files
import os
import math
import time
import pickle
import numpy as np
import xarray as xr
import argparse

from tqdm import tqdm

# Create splits on decomposed data as dictionaries of indices
def split_data_val_decomp(datasrc='./data/', filestr='uh25max', outdir='./data',
               n_class=20, class_sep=0.2, split=0.4, domain_gap=1.0, val_sz=0.2, max_val=75):
    # Parameters:
    # datasrc = source directory
    # filestr = filename for list of max uh25 values; expects $(filestr).csv
    # outdir = output directory
    # n_class = number of classes to generate when binning
    # class_sep = separation between classes, i.e. fraction that gets dropped of each class
    # split = train/test split
    # domain_gap = decides how train/test classes are selected; 1.0 = max domain gap, 0.0 = min domain gap
    # val_sz = fraction of training set to set aside for validation
    # max_val = maximum value used to calculate class sizes during binning
        # NOTE: data where max UH exceeds max_val will still be placed in the last class

    # Calculate class size
    class_size = max_val / n_class
    print("Max value: {:.2f} | Class size: {:.2f} | With separation: {:.2f}"
          .format(max_val, class_size, class_size*(1-class_sep)))

    # Bin classes

    # Calculate number of (source) classes to populate w/ high domain gap and low domain gap
    high_cls = round(n_class * domain_gap * split)
    low_cls = round(n_class * split) - high_cls

    # Populate high gap classes
    source_cls = [x for x in range(high_cls)]

    # Populate low gap classes
    for i in range(low_cls):
        source_cls.append(round(i * (n_class - high_cls) / low_cls) + high_cls)

    # Populate target classes w/ all classes not in source task
    target_cls = [x for x in range(n_class) if x not in source_cls]
    
    # Sample val_cls source_cls, then remove val_cls from source_cls
    # High gap val
    val_cls = [source_cls[i] for i in range(round(-val_sz*domain_gap*len(source_cls)),0)]
    # Low gap val
    low_val_sz = round(val_sz*len(source_cls)) - len(val_cls)
    
    source_cls = [x for x in source_cls if x not in val_cls]
    for i in range(1,low_val_sz+1):
        val_cls.append(source_cls[math.ceil(i * len(source_cls)/ (low_val_sz+1.0))])

    # Remove all val_cls from source_cls
    source_cls = [x for x in source_cls if x not in val_cls]
    
    val_cls.sort()
    
    print("source_cls:", len(source_cls), source_cls)
    print("val_cls   :", len(val_cls), val_cls)
    print("target_cls:", len(target_cls), target_cls)

    count_class = [0] * n_class
    timestamp = str(time.time())

    # Open label list
    if os.path.isfile(datasrc + filestr + '.csv'):
        uh25mx = []
        with open(datasrc + filestr + '.csv', 'rb') as f:
            uh25mx = np.loadtxt(f, delimiter=',')

        # Create list of classes; indices of uh_class must match patch indices
        uh_class = []
        for val in uh25mx:
            # Actual class size = class_size * (1 - separation)
            val_class = math.floor(val / class_size)

            # If calculated class exceeds number of classes, put in highest class
            if val_class >= n_class:
                val_class = n_class - 1
                print(val) # Print value to see how far it is above max_val

            # Remove the bottom (separation*100)% of data within this class
            # Note: a consequence of this is that all very light storms are pruned
            if val - (class_size * val_class) < class_size * class_sep:
                uh_class.append(-1)
            else:
                uh_class.append(val_class)
                count_class[val_class] += 1

        print("Samples per class: ", count_class)

        # Populate dictionaries
        src_dict = {}
        val_dict = {}
        tgt_dict = {}
        # Keys = indices, values = labels
        for i in range(len(uh_class)):
            if uh_class[i] in source_cls:
                src_dict.update({i: uh_class[i]})
            elif uh_class[i] in val_cls:
                val_dict.update({i: uh_class[i]})
            elif uh_class[i] in target_cls:
                tgt_dict.update({i: uh_class[i]})
        
        # Write serialized dictionaries to disk
        with open(outdir + "src_split_" + timestamp + '.pickle','wb') as f:
            pickle.dump(src_dict, f)
        with open(outdir + "val_split_" + timestamp + '.pickle','wb') as f:
            pickle.dump(val_dict, f)
        with open(outdir + "tgt_split_" + timestamp + '.pickle','wb') as f:
            pickle.dump(tgt_dict, f)

        print("Dataset split with validation complete. Dictionaries stored at",  outdir)
        print("Timestamp:", timestamp)


# Deprecated; this approach is not scalable and requires enough RAM to hold at least one split at a time
def split_data_val_multi(datasrc='./data/', filestr='spatial_storm_data_part', outdir='./data',
               n_class=20, class_sep=0.2, split=0.4, domain_gap=1.0, val_sz=0.2, max_val=75):
    # Parameters:
    # datasrc = source directory
    # filestr = part of filename; filenames are expected to follow '$(filestr)X.nc' for X in (1,2,...,n)
    # n_class = number of classes to generate when binning
    # class_sep = separation between classes, i.e. fraction that gets dropped of each class
    # split = train/test split
    # domain_gap = decides how train/test classes are selected; 1.0 = max domain gap, 0.0 = min domain gap
    # val_sz = fraction of training set to set aside for validation
    # max_val = maximum value used to calculate class sizes during binning
        # NOTE: data where max UH exceeds max_val will still be placed in the last class

    # Calculate class size
    class_size = max_val / n_class
    print("Max value: {:.2f} | Class size: {:.2f} | With separation: {:.2f}"
          .format(max_val, class_size, class_size*(1-class_sep)))

    # Bin classes

    # Calculate number of (source) classes to populate w/ high domain gap and low domain gap
    high_cls = round(n_class * domain_gap * split)
    low_cls = round(n_class * split) - high_cls

    # Populate high gap classes
    source_cls = [x for x in range(high_cls)]

    # Populate low gap classes
    for i in range(low_cls):
        source_cls.append(round(i * (n_class - high_cls) / low_cls) + high_cls)

    # Populate target classes w/ all classes not in source task
    target_cls = [x for x in range(n_class) if x not in source_cls]
    
    # Sample val_cls source_cls, then remove val_cls from source_cls
    # High gap val
    val_cls = [source_cls[i] for i in range(round(-val_sz*domain_gap*len(source_cls)),0)]
    # Low gap val
    low_val_sz = round(val_sz*len(source_cls)) - len(val_cls)
    
    source_cls = [x for x in source_cls if x not in val_cls]
    for i in range(1,low_val_sz+1):
        val_cls.append(source_cls[math.ceil(i * len(source_cls)/ (low_val_sz+1.0))])

    # Remove all val_cls from source_cls
    source_cls = [x for x in source_cls if x not in val_cls]
    
    val_cls.sort()
    
    print("source_cls:", len(source_cls), source_cls)
    print("val_cls   :", len(val_cls), val_cls)
    print("target_cls:", len(target_cls), target_cls)

    count_class = [0] * n_class
    timestamp = str(time.time()) # Calculate timestamp here, reuse throughout

    filenum = 1
    # Iterate through all files
    while os.path.isfile(datasrc + filestr + str(filenum) + '.nc'):
        print("Opening file", filenum)

        # Open dataset
        ds = xr.open_dataset(datasrc + filestr + str(filenum) + '.nc')

        # Calculate max UH for each sample then assign class
        uh25mx = ds['uh25'].max(['y','x'],skipna=True).values
        uh_class = []
        for val in uh25mx:
            # Actual class size = class_size * (1 - separation)
            val_class = math.floor(val / class_size)

            # If calculated class exceeds number of classes, put in highest class
            if val_class >= n_class:
                val_class = n_class - 1

            # Remove the bottom (separation*100)% of data within this class
            # Note: a consequence of this is that all very light storms are pruned
            if val - (class_size * val_class) < class_size * class_sep:
                uh_class.append(-1)
            else:
                uh_class.append(val_class)
                count_class[val_class] += 1
        
        print("Samples per class @", filenum, "=", count_class)
        
        # Generate list of patch indices to drop for src, tgt, val
        src_idx = []
        val_idx = []
        tgt_idx = []
        src_labels = []
        val_labels = []
        tgt_labels = []
        for i in range(len(uh_class)):
            if uh_class[i] in source_cls:
                src_idx.append(i)
                src_labels.append(uh_class[i])
            elif uh_class[i] in target_cls:
                tgt_idx.append(i)
                tgt_labels.append(uh_class[i])
            elif uh_class[i] in val_cls:
                val_idx.append(i)
                val_labels.append(uh_class[i])
        
        # For some reason, this converts the uh_cls column from int32 to float64
        source_task = ds.isel(patch=src_idx)
        val_task = ds.isel(patch=val_idx)
        target_task = ds.isel(patch=tgt_idx)
        ds.close()
        
        print("Writing to disk.")

        # Store label arrays
        with open(outdir + "src_labels_" + timestamp + ".csv","ab") as f:
            np.savetxt(f, src_labels, delimiter=',')
        with open(outdir + "val_labels_" + timestamp + ".csv","ab") as f:
            np.savetxt(f, val_labels, delimiter=',')
        with open(outdir + "tgt_labels_" + timestamp + ".csv","ab") as f:
            np.savetxt(f, tgt_labels, delimiter=',')
        
        # Drop labels from tasks
        source_task = source_task.drop_vars('uh25')
        val_task = val_task.drop_vars('uh25')
        target_task = target_task.drop_vars('uh25')

        # Dims are [patch, vars=23, x=32, y=32]
        src_data = np.stack([data_array.values for data_array in source_task.data_vars.values()], axis=3)
        source_task.close()
        
        val_data = np.stack([data_array.values for data_array in val_task.data_vars.values()], axis=3)
        val_task.close()
        
        tgt_data = np.stack([data_array.values for data_array in target_task.data_vars.values()], axis=3)
        target_task.close()

        # Store pickled data, add timestamp to avoid overwriting
        with open(outdir + "src_data_" + timestamp + '.pickle',"ab") as f:
            pickle.dump(src_data, f)
        with open(outdir + "val_data_" + timestamp + '.pickle',"ab") as f:
            pickle.dump(val_data, f)
        with open(outdir + "tgt_data_" + timestamp + '.pickle',"ab") as f:
            pickle.dump(tgt_data, f)
        
        # Move to next file
        filenum += 1
        
    print("Dataset split with validation complete. Files stored at",  outdir)
    print("Timestamp:", timestamp)

# Parse command line args
parser = argparse.ArgumentParser(description='Split dataset.')
parser.add_argument('--datasrc',    type=str,   default='./data/',  help='set datset source directory')
parser.add_argument('--filestr',    type=str,   default='uh25max',  help='set filename substring (expects $(filestr)X.nc, X = 1..n)')
parser.add_argument('--outdir',     type=str,   default='./data/',  help='directory to store output files')
parser.add_argument('--n_class',    type=int,   default=25,         help='number of classes to create when binning')
parser.add_argument('--max_val',    type=int,   default=75,         help='max value used when binning classes')
parser.add_argument('--class_sep',  type=float, default=0.2,        help='separation between classes')
parser.add_argument('--split',      type=float, default=0.4,        help='train-test split')
parser.add_argument('--val_sz',     type=float, default=0.2,        help='validation split')
parser.add_argument('--domain_gap', type=float, default=1.0,        help='domain gap, i.e. difference between splits; 0.0 is lowest, 1.0 is highest')

args = parser.parse_args()

# Run split function
split_data_val_decomp(datasrc=args.datasrc, filestr=args.filestr, outdir=args.outdir, 
                      n_class=args.n_class, max_val=args.max_val, class_sep=args.class_sep,
                      split=args.split, val_sz=args.val_sz, domain_gap=args.domain_gap)