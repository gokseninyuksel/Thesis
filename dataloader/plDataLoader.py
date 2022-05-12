from dataloader.LazyDataset import LazyDataset
import pytorch_lightning as pl
from torch.utils.data import DataLoader 
import torch 
from utils.utils import seed_worker
class LazyDataModule(pl.LightningDataModule):
    def __init__(self, data_dir_train, data_dir_val, data_dir_test, sources, mode, batch_size: int = 32 , num_workers = 4):
        super().__init__()
        self.data_dir_val = data_dir_val 
        self.data_dir_test = data_dir_test 
        self.data_dir_train = data_dir_train 
        self.sources = sources
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.generator = torch.Generator()
        self.generator.manual_seed(0)

    def setup(self,stage):
        self.test = LazyDataset(path = self.data_dir_test, is_train = False, sources = self.sources, mode = self.mode )
        self.val = LazyDataset(path = self.data_dir_val, is_train = False, sources = self.sources, mode = self.mode )
        self.train = LazyDataset(path = self.data_dir_train, is_train = True, sources = self.sources, mode = self.mode )

    def train_dataloader(self):
        return DataLoader(self.train, 
                          batch_size=self.batch_size,
                          num_workers = self.num_workers,
                          shuffle = True,
                          pin_memory = True,
                          worker_init_fn=seed_worker,
                          generator=self.generator)

    def val_dataloader(self):
        return DataLoader(self.val, 
                          batch_size=self.batch_size, 
                          num_workers = self.num_workers,
                          shuffle = False,
                          pin_memory = True,
                          worker_init_fn=seed_worker,
                          generator=self.generator)

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          num_workers = self.num_workers,
                          shuffle = False,
                          pin_memory = True,
                          worker_init_fn=seed_worker,
                          generator=self.generator)