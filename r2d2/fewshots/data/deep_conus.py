#import os
import pickle
#from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
# If you need to make this a torchnet Dataset instead, uncomment the below line and comment the above line
#from torchnet.dataset.dataset import Dataset
import torchvision.transforms as transforms
from torchvision import ops

# Imported from few-shot-meta-baseline

class DeepConus(Dataset):
    
    def __init__(self, root_path, split='train', **kwargs):
        # TODO: make this a parameter
        timestamp = '1669064393.6722744'
        
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
        
        self.n_classes = len(set(self.label))
        
        # Still need this
        label_lst = list(set(self.label))
        self.label = [label_lst.index(label) for label in self.label]
        
        # Unpickle data, store in self.data
        """
        self.data = None
        with open(datafile, 'rb') as f:
            _arr = []
            while True:
                try:
                    _arr = pickle.load(f)
                except EOFError:
                    break
                else:
                    if self.data is None:
                        self.data = _arr
                    else:
                        self.data = np.concatenate((self.data, _arr), axis=0)
        """
        # Transforms
        # Normalization (params generated in deep-conus-master/fewshot.ipynb)
        norm_params = {'mean': [280.8821716308594, 271.5213928222656, 260.1457214355469, 246.7049102783203, 8.42071533203125, 13.114259719848633, 16.928213119506836, 19.719449996948242, 6.177618026733398, 13.898662567138672, 18.913000106811523, 23.985916137695312, 0.007207642309367657, 0.0046530915424227715, 0.002190731931477785, 0.0007718075066804886, 868.15625, 678.8226928710938, 525.4044799804688, 401.36004638671875, 0.40490102767944336, 23.232492446899414, 8.562521934509277],
                       'std': [109.22666931152344, 109.22666931152344, 77.23491668701172, 109.22666931152344, 8.866754531860352, 9.56382942199707, 10.957494735717773, 11.892759323120117, 8.308595657348633, 9.732820510864258, 11.696307182312012, 14.249922752380371, 0.004077681340277195, 0.0025500282645225525, 0.0013640702236443758, 0.0005331166321411729, 309.60260009765625, 308.9396667480469, 195.89791870117188, 154.46983337402344, 0.48534637689590454, 15.682641983032227, 6.017237186431885]}
        normalize = transforms.Normalize(**norm_params)
        
        self.default_transform = transforms.Compose([transforms.ToTensor(), ops.Permute([1,2,0]), normalize])
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
        # Open patch file
        with open(self.root_path + 'patch_' + str(self.data[idx]) + '.pickle','rb') as f:
            return self.transform(pickle.load(f)), self.label[idx]
