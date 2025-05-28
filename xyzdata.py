import ase
import os
import pandas as pd
import numpy as np
import torch
import h5py
import lightning as L

#loader to import dicts:
import cace
from cace.data.atomic_data import AtomicData
from cace.tools.torch_geometric import Dataset, DataLoader

def from_xyz(a,cutoff,data_key):
    # a.info = {k:v for k,v in a.info.items() if k in ks}
    # a.arrays = {k:v for k,v in a.arrays.items() if k in ks}
    ad = AtomicData.from_atoms(a,cutoff=cutoff,data_key=data_key)
    if "c6" in ad.keys:
        ad["c6"] = ad["c6"].ravel()
        ad["c6_ptr"] = ad["c6"].shape[0]
    return ad

class XYZDataset(Dataset):
    def __init__(self,root_xyz,cutoff=4.0, drop_last=True,
                transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root_xyz, transform, pre_transform, pre_filter)
        self.root = root_xyz
        self.cutoff = cutoff
        self.prepare_data()
    
    def prepare_data(self):
        dataset = cace.tasks.get_dataset_from_xyz(self.root,valid_fraction=1e-10,cutoff=self.cutoff)
        data_key = [k for k,v in dataset.train[0].arrays.items() if (k not in ["numbers"])]
        data_key += [k for k,v in dataset.train[0].info.items() if (type(v) != str)]
        ks = ["energy","force","forces","c6","numbers","positions"]
        data_key = {k : k for k in ks}
        dataset = [from_xyz(a,self.cutoff,data_key) for a in dataset.train]
        self.dataset = dataset

    def len(self):
        return len(self.dataset)
    
    def get(self, idx):
        return self.dataset[idx]

class XYZData(L.LightningDataModule):
    def __init__(self, root_xyz, cutoff=4.0, batch_size=32, drop_last=True, shuffle=True, valid_p=0.05, test_p=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.root = root_xyz
        self.valid_p = valid_p
        self.test_p = test_p
        self.cutoff = cutoff
        self.drop_last = drop_last
        self.shuffle = shuffle
        try:
            self.num_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        except:
            self.num_cpus = os.cpu_count()
        self.prepare_data()
    
    def prepare_data(self):
        dataset = XYZDataset(self.root)
        torch.manual_seed(12345)
        if self.shuffle:
            dataset = dataset.shuffle()
        cut1 = int(len(dataset)*(1-self.valid_p-self.test_p))
        cut2 = int(len(dataset)*(1-self.test_p))
        self.train = dataset[:cut1]
        self.val = dataset[cut1:cut2]
        self.test = dataset[cut2:]

    def train_dataloader(self):
        train_loader = DataLoader(self.train, batch_size=self.batch_size, drop_last=self.drop_last,
                                  shuffle=True, num_workers = self.num_cpus)
        return train_loader

    def val_dataloader(self):
        val_loader = DataLoader(self.val, batch_size=self.batch_size, drop_last=False,
                                shuffle=False, num_workers = self.num_cpus)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test, batch_size=self.batch_size, drop_last=False,
                                shuffle=False, num_workers = self.num_cpus)
        return test_loader
