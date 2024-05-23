# Split function with validation, future storms as test data
import os
import math
import time
import pickle
import numpy as np
import xarray as xr
import argparse
import random

# Create splits on decomposed data as dictionaries of indices
def split_data_future(args):
    # NOTE: data where max UH exceeds max_val will still be placed in the last class

    # get variables from args
    datasrc = args.datasrc
    futuresrc = args.futuresrc
    filestr = args.filestr
    n_class = args.n_class
    n_class_future = args.n_class_future
    sample_future = args.sample_future
    class_sep = args.class_sep
    domain_gap = args.domain_gap
    val_sz = args.val_sz
    max_val = args.max_val
    exp = args.exp

    # Calculate (past storm) class size
    class_size = max_val / n_class
    print("Max value: {:.2f} | Class size: {:.2f} | With separation: {:.2f}"
          .format(max_val, class_size, class_size*(1-class_sep)))

    # Define class function: get floor from class number
    def class_val(n, mx, n_c):
        return mx * ((n / n_c) ** exp)

    # Get (unfloored) class number from current value
    def class_num(val, mx, n_c):
        return n_c * ((val / mx) ** (1/exp))

    # Bin classes

    source_cls = range(n_class)
    target_cls = range(n_class_future)

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
    print("target_cls:", len(target_cls), list(target_cls))

    count_class = [0] * n_class
    timestamp = str(time.time())

    # Open label list
    past_path = os.path.join(datasrc, filestr + '.csv')
    future_path = os.path.join(futuresrc, filestr + '.csv')
    if os.path.isfile(past_path) and os.path.isfile(future_path):
        # Load training set
        uh25mx = []
        with open(past_path, 'r') as f:
            uh25mx = np.loadtxt(f, delimiter=',')

        # Create list of classes; indices of uh_class must match patch indices
        uh_class = []
        above_max = 0
        for val in uh25mx:
            # Actual class size = class_size * (1 - separation)
            val_class = class_num(val, max_val, n_class)

            c = math.floor(val_class)
            # If calculated class exceeds number of classes, put in highest class
            if c >= n_class:
                c = n_class - 1
                above_max += 1

            # Remove the bottom (separation*100)% of data within this class
            # Note: a consequence of this is that all very light storms are pruned
            if class_val(val_class, max_val, n_class) < class_val(c + class_sep, max_val, n_class):
                uh_class.append(-1)
            else:
                uh_class.append(c)
                count_class[c] += 1

        print("Past storm samples per class: ", count_class)
        print("Samples above max:", above_max)

        # Populate dictionaries
        src_dict = {}
        val_dict = {}
        # Keys = indices, values = labels
        for i in range(len(uh_class)):
            if uh_class[i] in source_cls:
                src_dict.update({i: uh_class[i]})
            elif uh_class[i] in val_cls:
                val_dict.update({i: uh_class[i]})

        # Load test set
        uh25mx_test = []
        with open(future_path, 'r') as f:
            uh25mx_test = np.loadtxt(f, delimiter=',')

        # Calculate future max and class size
        future_max = max(uh25mx_test)
        future_class_size = future_max / n_class_future

        tgt_dict = {}
        count_class = [0] * n_class_future

        # subsample future storms so we don't have 1.4m test samples
        test_samples = len(uh25mx_test)
        for i in random.sample(range(test_samples), int(test_samples * sample_future)):
            val = uh25mx_test[i]
            # Actual class size = class_size * (1 - separation)
            val_class = math.floor(class_num(val, future_max, n_class_future))

            # If calculated class exceeds number of classes, put in highest class (this shouldn't happen)
            if val_class >= n_class_future:
                val_class = n_class_future - 1
                print(val) # Print value to see how far it is above max_val

            tgt_dict.update({i: val_class})
            count_class[val_class] += 1

        print("Future storm samples per class: ", count_class)
        print("Past max:", max(uh25mx), "| Future max:", max(uh25mx_test))

        print('Train samples:', len(src_dict))
        print('Val samples:', len(val_dict))
        print('Test samples:', len(tgt_dict))

        # Write serialized dictionaries to disk
        with open(os.path.join(datasrc, "src_split_" + timestamp + '.pickle'),'wb') as f:
            pickle.dump(src_dict, f)
        with open(os.path.join(datasrc, "val_split_" + timestamp + '.pickle'),'wb') as f:
            pickle.dump(val_dict, f)
        with open(os.path.join(futuresrc, "tgt_split_" + timestamp + '.pickle'),'wb') as f:
            pickle.dump(tgt_dict, f)

        print("Dataset split with validation complete. Past storms dictionary stored at", datasrc, "and future storms at", futuresrc)
        print("Timestamp:", timestamp)
    else:
        print("Uh25max files not found. Run decompose.py on past storms and future storms, each in different directories.")
        print("(Default: ./data/ and ./data-future/)")

# Parse command line args
parser = argparse.ArgumentParser(description='Split dataset.')
parser.add_argument('--datasrc',        type=str,   default='./data/',          help='set (past storm) datset source directory')
parser.add_argument('--futuresrc',      type=str,   default='./data-future/',   help='set future storm dataset source directory')
parser.add_argument('--filestr',        type=str,   default='uh25max',          help='set directory filename (expects $(filestr).csv)')
parser.add_argument('--n_class',        type=int,   default=15,                 help='number of classes to create when binning past storms')
parser.add_argument('--n_class_future', type=int,   default=5,                  help='number of classes to create when binning')
parser.add_argument('--sample_future',  type=float, default=0.01,               help='fraction of future storms to sample')
parser.add_argument('--max_val',        type=float, default=75.0,               help='max value used when binning classes')
parser.add_argument('--class_sep',      type=float, default=0.5,                help='separation between classes')
parser.add_argument('--val_sz',         type=float, default=0.35,               help='validation split')
parser.add_argument('--domain_gap',     type=float, default=1.0,                help='domain gap, i.e. difference between splits; 0.0 is lowest, 1.0 is highest')
parser.add_argument('--exp',            type=float, default=1.0,                help='binning exponent, e.g. 2.0 = quadratic')

args = parser.parse_args()

# Run split function
split_data_future(args)