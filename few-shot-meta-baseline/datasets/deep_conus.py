import pickle
import numpy as np
#import xarray as xr

import torch
from torch.utils.data import Dataset
from torchvision import transforms, ops

from .datasets import register

@register('deep-conus')
class DeepConus(Dataset):
    
    def __init__(self, root_path, timestamp, split='train', features='alt', **kwargs):
        # accepted splits: train, test, val
        # accepted features settings: full, alt

        timestamp = str(timestamp)
        splitfile = root_path
        self.root_path = root_path
        
        if split == 'train':
            splitfile += 'src_split_' + timestamp + '.pickle'
        elif split == 'test':
            splitfile += 'tgt_split_' + timestamp + '.pickle'
        elif split == 'val':
            splitfile += 'val_split_' + timestamp + '.pickle'
            
        else:
            print("Split options: train, test, val")
            return -1
        
        # Store list of labels + number of classes contained within it
        with open(splitfile, 'rb') as f:
            _dict = pickle.load(f)
            self.label = list(_dict.values())
            self.data = list(_dict.keys())
        
        # Trying using n_classes = max label instead of this
        label_set = set(self.label)
        self.n_classes = len(label_set)
        
        # Cross entropy doesn't work unless each label is in the range [0,n_classes]
        label_lst = list(label_set)
        self.label = [label_lst.index(label) for label in self.label]
        
        # Transforms
        # Normalization (params generated in deep-conus-master/fewshot.ipynb)
        norm_params = {'mean': [280.8821716308594, 271.5213928222656, 260.1457214355469, 246.7049102783203, 8.42071533203125, 13.114259719848633, 16.928213119506836, 19.719449996948242, 6.177618026733398, 13.898662567138672, 18.913000106811523, 23.985916137695312, 0.007207642309367657, 0.0046530915424227715, 0.002190731931477785, 0.0007718075066804886, 868.15625, 678.8226928710938, 525.4044799804688, 401.36004638671875, 0.40490102767944336, 23.232492446899414, 8.562521934509277],
                    'std': [109.22666931152344, 109.22666931152344, 77.23491668701172, 109.22666931152344, 8.866754531860352, 9.56382942199707, 10.957494735717773, 11.892759323120117, 8.308595657348633, 9.732820510864258, 11.696307182312012, 14.249922752380371, 0.004077681340277195, 0.0025500282645225525, 0.0013640702236443758, 0.0005331166321411729, 309.60260009765625, 308.9396667480469, 195.89791870117188, 154.46983337402344, 0.48534637689590454, 15.682641983032227, 6.017237186431885]}
        if features == 'full':
            n_channels = 23
            normalize = transforms.Normalize(**norm_params)
            tlst = [transforms.ToTensor(), ops.Permute([1,2,0]), normalize]
        elif features == 'alt':
            n_channels = 16

            # remove unwanted channels from norm_params
            norm_params['mean'] = norm_params['mean'][:12] + norm_params['mean'][16:20]
            norm_params['std'] = norm_params['std'][:12] + norm_params['std'][16:20]

            # lambda to remove the same channels from each tensor
            channels = torch.concat([torch.arange(0, 12), torch.arange(16, 20)])
            drop_channels = transforms.Lambda(lambda x: x[channels, ...])

            normalize = transforms.Normalize(**norm_params)
            tlst = [transforms.ToTensor(), ops.Permute([1,2,0]), drop_channels, normalize]
        
        self.default_transform = transforms.Compose(tlst)
        self.transform = self.default_transform
        
        # Convert from normalized to raw
        def convert_raw(x):
            mean = torch.tensor(norm_params['mean']).view(n_channels, 1, 1).type_as(x)
            std = torch.tensor(norm_params['std']).view(n_channels, 1, 1).type_as(x)
            return x * std + mean
        
        self.convert_raw = convert_raw

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # Open patch file
        with open(self.root_path + 'patch_' + str(self.data[idx]) + '.pickle','rb') as f:
            return self.transform(pickle.load(f)), self.label[idx]