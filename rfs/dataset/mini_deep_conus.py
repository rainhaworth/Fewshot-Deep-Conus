import os
import pickle
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

# Need to like, register this somewhere

class DeepConus(Dataset):
    def __init__(self, args, partition='train', pretrain=True, is_sample=False, k=4096,
                 transform=None):
        super(Dataset, self).__init__()
        self.data_root = args.data_root
        self.partition = partition
        self.data_aug = args.data_aug
        # Maybe I should handle these differently lol
        # Do we need to convert_raw?
        self.mean = [280.8821716308594, 271.5213928222656, 260.1457214355469, 246.7049102783203, 8.42071533203125, 13.114259719848633, 16.928213119506836, 19.719449996948242, 6.177618026733398, 13.898662567138672, 18.913000106811523, 23.985916137695312, 0.007207642309367657, 0.0046530915424227715, 0.002190731931477785, 0.0007718075066804886, 868.15625, 678.8226928710938, 525.4044799804688, 401.36004638671875, 0.40490102767944336, 23.232492446899414, 8.562521934509277]
        self.std = [109.22666931152344, 109.22666931152344, 77.23491668701172, 109.22666931152344, 8.866754531860352, 9.56382942199707, 10.957494735717773, 11.892759323120117, 8.308595657348633, 9.732820510864258, 11.696307182312012, 14.249922752380371, 0.004077681340277195, 0.0025500282645225525, 0.0013640702236443758, 0.0005331166321411729, 309.60260009765625, 308.9396667480469, 195.89791870117188, 154.46983337402344, 0.48534637689590454, 15.682641983032227, 6.017237186431885]
        self.normalize = transforms.Normalize(mean=self.mean, std=self.std)
        self.pretrain = pretrain
        
        self.transform = transforms.Compose([transforms.ToTensor(), self.normalize])

        # Notes on args:
            # We probably have to keep is_sample = false, which means --n_aug_support_samples 0 for meta learning

        timestamp = '1666844475.2518342'
        
        datafile = self.data_root
        labelfile = self.data_root
        
        if partition == 'train':
            datafile += 'src_data_' + timestamp + '.pickle'
            labelfile += 'src_labels_' + timestamp + '.csv'
            
        elif partition == 'test':
            datafile += 'tgt_data_' + timestamp + '.pickle'
            labelfile += 'tgt_labels_' + timestamp + '.csv'
            
        elif partition == 'val':
            datafile += 'val_data_' + timestamp + '.pickle'
            labelfile += 'val_labels_' + timestamp + '.csv'
            
        else:
            print("Partition options: train, test, val")
            return -1
        
        # Store list of labels + number of classes contained within it
        self.labels = np.loadtxt(labelfile, delimiter=',')
        self.labels = self.labels.astype(np.int64) # Convert to long
        
        self.n_classes = len(set(self.labels))
        
        # Cross entropy doesn't work unless each label is in the range [0,n_classes]
        label_lst = list(set(self.labels))
        self.labels = [label_lst.index(label) for label in self.labels]
        
        # Unpickle data, store in self.imgs
        pickle_in = open(datafile, 'rb')
        self.data = pickle.load(pickle_in)
        self.imgs = self.data
        
        # idk what pretrain means, or any of the rest of this for that matter
        # just commenting everything out for now
        """
        if self.pretrain:
            self.file_pattern = 'miniImageNet_category_split_train_phase_%s.pickle'           
        else:
            self.file_pattern = 'miniImageNet_category_split_%s.pickle' 
        
        #self.data = {}
        with open(os.path.join(self.data_root, self.file_pattern % partition), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.imgs = data['data']
            self.labels = data['labels']

        # pre-process for contrastive sampling
        # that comment^ is from them, idk what any of this means but i doubt it helps with non-images
        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            self.labels = np.asarray(self.labels)
            self.labels = self.labels - np.min(self.labels)
            num_classes = np.max(self.labels) + 1

            self.cls_positive = [[] for _ in range(num_classes)]
            for i in range(len(self.imgs)):
                self.cls_positive[self.labels[i]].append(i)

            self.cls_negative = [[] for _ in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i]) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i]) for i in range(num_classes)]
            self.cls_positive = np.asarray(self.cls_positive)
            self.cls_negative = np.asarray(self.cls_negative)
            """

    def __getitem__(self, item):
        # For some reason we need to return the index they passed in
        return self.transform(self.data[item]), self.labels[item], item
        
        # once again not really sure what's going on, commenting out
        """
        img = np.asarray(self.imgs[item]).astype('uint8')
        img = self.transform(img)
        target = self.labels[item] - min(self.labels)

        if not self.is_sample:
            return img, target, item
        else:
            pos_idx = item
            replace = True if self.k > len(self.cls_negative[target]) else False
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=replace)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, item, sample_idx
            """
        
    def __len__(self):
        return len(self.labels)


class MetaDeepConus(DeepConus):
    
    def __init__(self, args, partition='train', train_transform=None, test_transform=None, fix_seed=True):
        super(MetaDeepConus, self).__init__(args, partition, False)
        self.fix_seed = fix_seed
        self.n_ways = args.n_ways
        self.n_shots = args.n_shots
        self.n_queries = args.n_queries
        #self.classes = list(self.data.keys()) # This breaks things so I'm commenting it out
        self.n_test_runs = args.n_test_runs
        self.n_aug_support_samples = args.n_aug_support_samples
        
        # Comment out data augmentation
        """
        if train_transform is None:
            self.train_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.RandomCrop(84, padding=8),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                    transforms.RandomHorizontalFlip(),
                    lambda x: np.asarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.train_transform = train_transform

        if test_transform is None:
            self.test_transform = transforms.Compose([
                    lambda x: Image.fromarray(x),
                    transforms.ToTensor(),
                    self.normalize
                ])
        else:
            self.test_transform = test_transform
        """
        
        self.train_transform = transforms.ToTensor()
        self.test_transform = transforms.ToTensor()
        
        # I don't see the point of this but I guess it can't hurt??
        self.data = {}
        for idx in range(self.imgs.shape[0]):
            if self.labels[idx] not in self.data:
                self.data[self.labels[idx]] = []
            self.data[self.labels[idx]].append(self.imgs[idx])
        self.classes = list(self.data.keys())

    def __getitem__(self, item):
        # I'm cautiously optimistic that this part will Just Work
        # Might need to shuffle around axes?
        if self.fix_seed:
            np.random.seed(item)
        cls_sampled = np.random.choice(self.classes, self.n_ways, False)
        support_xs = []
        support_ys = []
        query_xs = []
        query_ys = []
        for idx, cls in enumerate(cls_sampled):
            imgs = np.asarray(self.data[cls]).astype('uint8')
            support_xs_ids_sampled = np.random.choice(range(imgs.shape[0]), self.n_shots, False)
            support_xs.append(imgs[support_xs_ids_sampled])
            support_ys.append([idx] * self.n_shots)
            query_xs_ids = np.setxor1d(np.arange(imgs.shape[0]), support_xs_ids_sampled)
            query_xs_ids = np.random.choice(query_xs_ids, self.n_queries, False)
            query_xs.append(imgs[query_xs_ids])
            query_ys.append([idx] * query_xs_ids.shape[0])
        support_xs, support_ys, query_xs, query_ys = np.array(support_xs), np.array(support_ys), np.array(
            query_xs), np.array(query_ys)
        num_ways, n_queries_per_way, height, width, channel = query_xs.shape
        query_xs = query_xs.reshape((num_ways * n_queries_per_way, height, width, channel))
        query_ys = query_ys.reshape((num_ways * n_queries_per_way, ))
                
        support_xs = support_xs.reshape((-1, height, width, channel))
        if self.n_aug_support_samples > 1:
            support_xs = np.tile(support_xs, (self.n_aug_support_samples, 1, 1, 1))
            support_ys = np.tile(support_ys.reshape((-1, )), (self.n_aug_support_samples))
        support_xs = np.split(support_xs, support_xs.shape[0], axis=0)
        query_xs = query_xs.reshape((-1, height, width, channel))
        query_xs = np.split(query_xs, query_xs.shape[0], axis=0)
        
        support_xs = torch.stack(list(map(lambda x: self.train_transform(x.squeeze()), support_xs)))
        query_xs = torch.stack(list(map(lambda x: self.test_transform(x.squeeze()), query_xs)))
      
        return support_xs, support_ys, query_xs, query_ys      
        
    def __len__(self):
        return self.n_test_runs
    
    
if __name__ == '__main__':
    args = lambda x: None
    args.n_ways = 5
    args.n_shots = 1
    args.n_queries = 12
    args.data_root = 'data'
    args.data_aug = True
    args.n_test_runs = 5
    args.n_aug_support_samples = 1
    deepconus = DeepConus(args, 'val')
    print(len(deepconus))
    print(deepconus.__getitem__(500)[0].shape)
    
    metadeepconus = MetaDeepConus(args)
    print(len(metadeepconus))
    print(metaimagenet.__getitem__(500)[0].size())
    print(metaimagenet.__getitem__(500)[1].shape)
    print(metaimagenet.__getitem__(500)[2].size())
    print(metaimagenet.__getitem__(500)[3].shape)