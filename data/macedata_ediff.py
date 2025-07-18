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

def from_config(a,c,cutoff,z_table,info_ks,array_ks):
    ad = AtomicData.from_config(c,cutoff=cutoff,z_table=z_table)
    dtype = ad["positions"].dtype
    for k in info_ks:
        if type(a.info[k]) == str:
            continue
        ad[k] = torch.tensor(a.info[k],dtype=dtype)
    for k in array_ks:
        if k == "c6":
            #Remove non-diagonal c6
            c6 = torch.tensor(a.arrays[k],dtype=ad["positions"].dtype)
            mask = ~torch.eye(c6.shape[0], dtype=torch.bool, device=c6.device)
            ad["c6"] = c6[mask].ravel()
            # ad["c6_ptr"] = ad["c6"].shape[0] Just n(n-1)
        else:
            ad[k] = torch.tensor(a.arrays[k],dtype=dtype)
    return ad

def from_config(a,c,cutoff,z_table,info_ks,array_ks):
    ad = AtomicData.from_config(c,cutoff=cutoff,z_table=z_table)
    dtype = ad["positions"].dtype
    for k in info_ks:
        if type(a.info[k]) == str:
            continue
        ad[k] = torch.tensor(a.info[k],dtype=dtype)
    for k in array_ks:
        ad[k] = torch.tensor(a.arrays[k],dtype=dtype)
    return ad

class MaceXYZDataset(Dataset):
    def __init__(self,dimer_xyz,monA_xyz,monB_xyz,cutoff=4.5,zs=[1,6,7,8,9,15,16,17,35,53],
                transform=None, pre_transform=None, pre_filter=None):
        super().__init__(dimer_xyz, transform, pre_transform, pre_filter)
        self.dimer_xyz = dimer_xyz
        self.monA_xyz = monA_xyz
        self.monB_xyz = monB_xyz
        self.cutoff = cutoff
        self.zs = zs
        self.prepare_data()
    
    def prepare_data(self):
        from mace.data import config_from_atoms_list
        from mace.tools.utils import AtomicNumberTable
        from ase import io
        import torch
        dimer_atms = ase.io.read(self.dimer_xyz,index=":")
        monA_atms = ase.io.read(self.monA_xyz,index=":")
        monB_atms = ase.io.read(self.monB_xyz,index=":")
        atm = dimer_atms[0]
        info_ks = atm.info.keys()
        array_ks = atm.arrays.keys()
        dimer_configs = config_from_atoms_list(dimer_atms)
        monA_configs = config_from_atoms_list(monA_atms)
        monB_configs = config_from_atoms_list(monB_atms)
        z_table = AtomicNumberTable(self.zs)
        dataset = [from_config(a,c,self.cutoff,z_table,info_ks,[]) for a,c in zip(dimer_atms,dimer_configs)]
        monA_dataset = [from_config(a,c,self.cutoff,z_table,[],[]) for a,c in zip(monA_atms,monA_configs)]
        monB_dataset = [from_config(a,c,self.cutoff,z_table,[],[]) for a,c in zip(monB_atms,monB_configs)]
        ks = ["positions","node_attrs","edge_index","shifts","unit_shifts"]
        # print(dataset[0].keys)
        for ad, ad_monA, ad_monB in zip(dataset,monA_dataset,monB_dataset):
            for k in ks:
                # if k not in ad_monA.keys:
                #     print("Missing",k)
                ad[f"{k}_monA"] = ad_monA[k]
                ad[f"{k}_monB"] = ad_monB[k]
            ad["numA"] = ad_monA["positions"].shape[0]
            ad["numB"] = ad_monB["positions"].shape[0]
            ad["numA_edges"] = ad_monA["edge_index"].shape[1]
            ad["numB_edges"] = ad_monB["edge_index"].shape[1]
            ad["numD"] = ad["positions"].shape[0]
        self.dataset = dataset

    def len(self):
        return len(self.dataset)
    
    def get(self, idx):
        return self.dataset[idx]

class MaceXYZData(L.LightningDataModule):
    def __init__(self, dimer_xyz, monA_xyz, monB_xyz, cutoff=4.5, batch_size=32,
                 drop_last=True, shuffle=True, valid_p=0.05, test_p=0.05):
        super().__init__()
        self.batch_size = batch_size
        self.dimer_xyz = dimer_xyz
        self.monA_xyz = monA_xyz
        self.monB_xyz = monB_xyz
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
        dataset = MaceXYZDataset(self.dimer_xyz,self.monA_xyz,self.monB_xyz)
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