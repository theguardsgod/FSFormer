'''The following module deals with creating the loader he'''
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
import numpy as np

from dataloader import Mydataset


class LoaderHelper:
    '''An abstract class for assisting with dataset creation.'''
    def __init__(self):

        self.dataset = Mydataset(dataPath="../Intrusiondatasmote.mat")

        

        self.indices = []
        self.set_indices()




    def set_indices(self, total_folds=5):
        '''Abstract function to set indices'''
        test_split = .2
        shuffle_dataset = True
        random_seed = 42

        dataset_size = len(self.dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(test_split * dataset_size))

        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        fold_indices = []
        lb_split = 0
        ub_split = split

        for _ in range(total_folds):
            train_indices = indices[:lb_split] + indices[ub_split:]
            test_indices = indices[lb_split:ub_split]
            lb_split = split + lb_split
            ub_split = split + ub_split
            fold_indices.append((train_indices, test_indices))

        self.indices = fold_indices

    
    def get_train_dl(self, fold_ind, shuffle=True):

        train_ds = Subset(self.dataset, self.indices[fold_ind][0])
        train_dl = DataLoader(train_ds, batch_size=64, shuffle=shuffle, num_workers=0, drop_last=True)

        return train_dl


    def get_test_dl(self, fold_ind, shuffle=True):

        test_ds = Subset(self.dataset, self.indices[fold_ind][1])
        test_dl = DataLoader(test_ds, batch_size=64, shuffle=shuffle, num_workers=0, drop_last=True)

        return test_dl



