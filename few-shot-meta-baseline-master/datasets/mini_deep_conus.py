import math
import pickle
import os
import numpy as np
#import xarray as xr

import torch
from torch.utils.data import Dataset
from torchvision import transforms # Only used to transform to tensor

from .datasets import register

@register('mini-deep-conus')
class MiniDeepConus(Dataset):
    
    def __init__(self, root_path, datasrc='../deep-conus-master/data/', split='train', **kwargs):
        # TODO: make this a parameter
        #timestamp = '1666844475.2518342'
        timestamp = '1666997683.8305142'
        
        datafile = datasrc
        labelfile = datasrc
        
        if split == 'train':
            datafile += 'src_data_' + timestamp + '.pickle'
            labelfile += 'src_labels_' + timestamp + '.csv'
            
        elif split == 'test':
            # Change it back to tgt
            datafile += 'tgt_data_' + timestamp + '.pickle'
            labelfile += 'tgt_labels_' + timestamp + '.csv'
            
        elif split == 'val':
            datafile += 'val_data_' + timestamp + '.pickle'
            labelfile += 'val_labels_' + timestamp + '.csv'
            
        else:
            print("Split options: train, test, val")
            return -1
        
        # Store list of labels + number of classes contained within it
        self.label = np.loadtxt(labelfile, delimiter=',')
        self.label = self.label.astype(np.int64) # Convert to long
        
        
        # Trying using n_classes = max label instead of this
        #self.n_classes = max(self.label) + 1
        self.n_classes = len(set(self.label))
        
        # Cross entropy doesn't work unless each label is in the range [0,n_classes]
        # So to achieve that, we have to do this (hopefully it won't cause weird issues)
        label_lst = list(set(self.label))
        self.label = [label_lst.index(label) for label in self.label]
        
        # Unpickle data, store in self.data
        pickle_in = open(datafile, 'rb')
        self.data = pickle.load(pickle_in)
        
        # Transforms
        # Normalization (params generated in deep-conus-master/fewshot.ipynb)
        norm_params = {'mean': [280.8821716308594, 271.5213928222656, 260.1457214355469, 246.7049102783203, 8.42071533203125, 13.114259719848633, 16.928213119506836, 19.719449996948242, 6.177618026733398, 13.898662567138672, 18.913000106811523, 23.985916137695312, 0.007207642309367657, 0.0046530915424227715, 0.002190731931477785, 0.0007718075066804886, 868.15625, 678.8226928710938, 525.4044799804688, 401.36004638671875, 0.40490102767944336, 23.232492446899414, 8.562521934509277],
                       'std': [109.22666931152344, 109.22666931152344, 77.23491668701172, 109.22666931152344, 8.866754531860352, 9.56382942199707, 10.957494735717773, 11.892759323120117, 8.308595657348633, 9.732820510864258, 11.696307182312012, 14.249922752380371, 0.004077681340277195, 0.0025500282645225525, 0.0013640702236443758, 0.0005331166321411729, 309.60260009765625, 308.9396667480469, 195.89791870117188, 154.46983337402344, 0.48534637689590454, 15.682641983032227, 6.017237186431885]}
        normalize = transforms.Normalize(**norm_params)
        
        self.default_transform = transforms.Compose([transforms.ToTensor(), normalize])
        self.transform = self.default_transform
        
        # Convert from normalized to raw
        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(23, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(23, 1, 1).type_as(x)
            return x * std + mean
        
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.transform(self.data[idx]), self.label[idx]