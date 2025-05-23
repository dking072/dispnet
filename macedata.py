import ase
import os
import pandas as pd
import numpy as np
import torch
import h5py
import lightning as L

#loader to import dicts:
from mace.data import AtomicData
from mace.tools.torch_geometric import Dataset, DataLoader
# from cace.tools.torch_geometric import Dataset, DataLoader

def from_config(a,c,cutoff,z_table):
    ks=["energy","force","forces","c6","numbers","positions"]
    # a.info = {k:v for k,v in a.info.items() if type(v) != str}
    ad = AtomicData.from_config(c,cutoff=cutoff,z_table=z_table)
    for k,v in a.info.items():
        if k not in ks:
            continue
        ad[k] = torch.tensor(v,dtype=ad["positions"].dtype)
    for k,v in a.arrays.items():
        if k not in ks:
            continue
        if k == "c6":
            ad["c6"] = torch.tensor(v,dtype=ad["positions"].dtype).ravel()
            ad["c6_ptr"] = ad["c6"].shape[0]
        else:
            ad[k] = torch.tensor(v,dtype=ad["positions"].dtype)
    return ad

class MaceXYZDataset(Dataset):
    def __init__(self,root_xyz,cutoff=4.0,zs=[1,6,7,8,9,15,16,17,35,53],ks=["c6"],
                transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root_xyz, transform, pre_transform, pre_filter)
        self.root = root_xyz
        self.cutoff = cutoff
        self.zs = zs
        self.ks = ks
        self.prepare_data()
    
    def prepare_data(self):
        from mace.data import config_from_atoms_list
        from mace.tools.utils import AtomicNumberTable
        from ase import io
        import torch
        atms = ase.io.read(self.root,index=":")
        configs = config_from_atoms_list(atms)
        z_table = AtomicNumberTable(self.zs)
        dataset = [from_config(a,c,self.cutoff,z_table) for a,c in zip(atms,configs)]
        self.dataset = dataset

    def len(self):
        return len(self.dataset)
    
    def get(self, idx):
        return self.dataset[idx]

class MaceXYZData(L.LightningDataModule):
    def __init__(self, root_xyz, cutoff=4.0, batch_size=32, drop_last=True, shuffle=True, valid_p=0.05, test_p=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.root = root_xyz
        self.valid_p = valid_p
        self.test_p = test_p
        self.cutoff = cutoff
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_cpus = 1
        # try:
        #     self.num_cpus = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
        # except:
        #     self.num_cpus = os.cpu_count()
        self.prepare_data()
    
    def prepare_data(self):
        dataset = MaceXYZDataset(self.root)
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