import os
import numpy as np
import torch
from functools import partial
from torchnet.transform import compose
from torchnet.dataset import ListDataset, TransformDataset

from fewshots.data.base import convert_dict, CudaTransform, EpisodicBatchSampler
from fewshots.data.setup import setup_images
from fewshots.data.cache import Cache
from fewshots.utils import filter_opt
from fewshots.data.SetupEpisode import SetupEpisode

from fewshots.data.mini_deep_conus import MiniDeepConus

from tqdm import tqdm

root_dir = ''

# load_setup_images but for deep conus
def load_deep_conus(cache, batch_size, augm_opt, d):
    cache_len = len(cache.data[d[1]])
    rand_ids = np.random.choice(cache_len, size=batch_size)
    out_dicts = [{'class': d[1], 'data': torch.cat([cache.data[d[1]][i] for i in rand_ids], dim=0)}]
    return out_dicts

def extract_episode(setup_episode, augm_opt, d):
    # data: N x C x H x W
    n_max_examples = d[0]['data'].size(0)

    n_way, n_shot, n_query = setup_episode.get_current_setup()
    
    example_inds = torch.randperm(n_max_examples)[:(n_shot + n_query)]

    support_inds = example_inds[:n_shot]
    query_inds = example_inds[n_shot:]

    xs_list = [d[i]['data'][support_inds] for i in range(augm_opt['n_augment'])]
    # concatenate as shots into xs
    xs = torch.cat(xs_list, dim=0)
    # extract queries from a single cache entry
    xq = d[np.random.randint(augm_opt['n_augment'])]['data'][query_inds]
    out_dict = {'class': d[0]['class'], 'xs': xs, 'xq': xq, 'n_way': n_way, 'n_shot': n_shot, 'n_query': n_query}
    return out_dict


def load_data(opt, splits):
    global root_dir
    root_dir = opt['data.root_dir']
    augm_opt = filter_opt(opt, 'augm')
    dataset = opt['data.dataset']
    split_dir = os.path.join(opt['data.root_dir'], opt['data.dataset'], 'splits', opt['data.split'])

    ret = {}
    # cache = {}
    cache = Cache()

    for split in splits:
        if split in ['val1', 'val5', 'test']:
            n_way = opt['data.test_way']
        else:
            n_way = opt['data.way']

        if split in ['train', 'trainval']:
            # random shots
            SE = SetupEpisode(batch_size=opt['data.batch_size'], shot_max=opt['data.shot_max'],
                              fixed_shot=opt['data.shot'], way_min=opt['data.way_min'], fixed_way=n_way)
        elif split == 'val1':
            SE = SetupEpisode(batch_size=opt['data.batch_size'], shot_max=opt['data.shot_max'], fixed_shot=1,
                              way_min=opt['data.way_min'], fixed_way=n_way)
        elif split == 'val5':
            SE = SetupEpisode(batch_size=opt['data.batch_size'], shot_max=opt['data.shot_max'], fixed_shot=5,
                              way_min=opt['data.way_min'], fixed_way=n_way)
        else:
            SE = SetupEpisode(batch_size=opt['data.batch_size'], shot_max=opt['data.shot_max'],
                              fixed_shot=opt['data.test_shot'], way_min=opt['data.way_min'], fixed_way=n_way)

        if split in ['val1', 'val5', 'test']:
            n_episodes = opt['data.test_episodes']
        else:
            n_episodes = opt['data.train_episodes']

        # Change split name
        splitname = split
        if split in ['val1','val5']:
            splitname = 'val'
        
        ds = MiniDeepConus(root_dir + '/', split=splitname)

        # populate Cache
        cache.data.update({0: []})
        _loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False)
        for sample in tqdm(_loader, desc='Loading ' + split + ' sample cache'):
            if sample[1].item() not in cache.data.keys():
                cache.data.update({sample[1].item(): []})
            cache.data[sample[1].item()].append(sample[0])

        # set batch size to minimum cache length
        # if this causes issues, set it manually
        batch_sz = min([len(cache.data[i]) for i in cache.data.keys()])

        # set max batch size
        #batch_sz = min(200, batch_sz)

        # set transforms
        transforms = [#partial(convert_dict, 'class'),
                      #partial(load_class_images, split, dataset, cache, augm_opt),
                      partial(load_deep_conus, cache, batch_sz, augm_opt), #added
                      partial(extract_episode, SE, augm_opt)]

        if opt['data.cuda']:
            transforms.append(CudaTransform())

        transforms = compose(transforms)

        tds = TransformDataset(ds, transforms)
        
        sampler = EpisodicBatchSampler(SE, len(tds), n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(tds, batch_sampler=sampler, num_workers=0)

    return ret


def load_class_images(split, dataset, cache, augm_opt, d):
    if d['class'] in cache.data.keys():
        if len(cache.data[d['class']]) < augm_opt['cache_size']:
            init_entry = False
            setup_images(split, d, cache, dataset, init_entry, root_dir, augm_opt)
    else:
        init_entry = True
        setup_images(split, d, cache, dataset, init_entry, root_dir, augm_opt)

    cache_len = len(cache.data[d['class']])

    # if cache does not enough shots yet, repeat
    if cache_len < augm_opt['n_augment']:
        rand_ids = np.random.choice(cache_len, size=augm_opt['n_augment'], replace=True)
    else:
        rand_ids = np.random.choice(cache_len, size=augm_opt['n_augment'], replace=False)

    out_dicts = [{'class': d['class'], 'data': cache.data[d['class']][rand_ids[i]]} for i in
                 range(augm_opt['n_augment'])]

    return out_dicts
