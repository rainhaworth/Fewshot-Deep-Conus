# From omniglot.py
# This should work with how I have it implemented now; data.split is unused

import os
import sys
import glob

from functools import partial

import pickle

import numpy as np
from PIL import Image

import torch
from torchvision.transforms import ToTensor, Normalize

from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

import protonets
from protonets.data.base import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler

DEEPCONUS_DATA_DIR = os.path.join(os.path.dirname(__file__), '../../../deep-conus/data')
DEEPCONUS_CACHE = { }

timestamp = "1669064393.6722744"

# CUSTOM FUNCTIONS: load_sample_path() and load_class_samples()
# TODO: add normalization?

def load_sample_path(key, out_field, d):
    # Open pickle file, assuming d[key] contains the correct filename
    with open(d[key],'rb') as f:
        d[out_field] = pickle.load(f)
    return d


norm_params = {'mean': [280.8821716308594, 271.5213928222656, 260.1457214355469, 246.7049102783203, 8.42071533203125, 13.114259719848633, 16.928213119506836, 19.719449996948242, 6.177618026733398, 13.898662567138672, 18.913000106811523, 23.985916137695312, 0.007207642309367657, 0.0046530915424227715, 0.002190731931477785, 0.0007718075066804886, 868.15625, 678.8226928710938, 525.4044799804688, 401.36004638671875, 0.40490102767944336, 23.232492446899414, 8.562521934509277],
                       'std': [109.22666931152344, 109.22666931152344, 77.23491668701172, 109.22666931152344, 8.866754531860352, 9.56382942199707, 10.957494735717773, 11.892759323120117, 8.308595657348633, 9.732820510864258, 11.696307182312012, 14.249922752380371, 0.004077681340277195, 0.0025500282645225525, 0.0013640702236443758, 0.0005331166321411729, 309.60260009765625, 308.9396667480469, 195.89791870117188, 154.46983337402344, 0.48534637689590454, 15.682641983032227, 6.017237186431885]}
normalize = Normalize(**norm_params)

def convert_tensor(key, d):
    # Old implementation that I don't understand:
    #d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].shape[0], d[key].shape[1])
    d[key] = torch.from_numpy(d[key])
    # add normalization
    d[key] = normalize.forward(d[key])
    return d

def load_class_samples(classes, ids, d):
    if d['class'] not in DEEPCONUS_CACHE:
        # Instead of parsing filename and grabbing all pngs in directory, grab from list of patches
        _samples = []
        # set hard limit so i can make this run faster
        limit = 1000
        for i in range(len(ids)):
            if classes[i] == d['class'] and len(_samples) < limit:
                _samples.append(ids[i])
        print("Loading class", d['class'], "with", len(_samples), "samples")
        class_samples = sorted([os.path.join(DEEPCONUS_DATA_DIR, 'patch_' + str(i) + '.pickle') for i in _samples])
        if len(class_samples) == 0:
            raise Exception("No samples found for deep conus class {} at {}.".format(d['class'], DEEPCONUS_DATA_DIR))

        # Open files w/ TransformDataset, load into cache
        image_ds = TransformDataset(ListDataset(class_samples),
                                    compose([partial(convert_dict, 'file_name'),
                                             partial(load_sample_path, 'file_name', 'data'),
                                             partial(convert_tensor, 'data')]))

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            DEEPCONUS_CACHE[d['class']] = sample['data']
            break # only need one sample because batch size equal to dataset length

    return { 'class': d['class'], 'data': DEEPCONUS_CACHE[d['class']] }

def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def load(opt, splits):
    #split_dir = os.path.join(OMNIGLOT_DATA_DIR, 'splits', opt['data.split'])

    ret = { }
    for split in splits:
        if split in ['val', 'test'] and opt['data.test_way'] != 0:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['val', 'test'] and opt['data.test_shot'] != 0:
            n_support = opt['data.test_shot']
        else:
            n_support = opt['data.shot']

        if split in ['val', 'test'] and opt['data.test_query'] != 0:
            n_query = opt['data.test_query']
        else:
            n_query = opt['data.query']

        if split in ['val', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        # TODO: change R2D2 to be more like this file and maybe fix it

        splitfile = ""
        if split == 'train':
            splitfile = 'src_split_' + timestamp + '.pickle'
        elif split == 'test':
            splitfile = 'tgt_split_' + timestamp + '.pickle'
        elif split == 'val':
            splitfile = 'val_split_' + timestamp + '.pickle'

        splitfile = os.path.join(DEEPCONUS_DATA_DIR, splitfile)

        class_names = []
        class_ids = []
        with open(splitfile, 'rb') as f:
            _dict = pickle.load(f)
            class_names = list(_dict.values())
            class_ids = list(_dict.keys())

        # Need to define transforms here so we can pass in the (disassembled) dictionary
        transforms = [partial(convert_dict, 'class'),
                      partial(load_class_samples, class_names, class_ids),
                      partial(extract_episode, n_support, n_query)]
        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        # Make dataset
        ds = TransformDataset(ListDataset(class_names), transforms)

        if opt['data.sequential']:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret
