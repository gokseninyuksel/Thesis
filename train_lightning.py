import json
from trainer import SVSGAN
from dataloader.plDataLoader import LazyDataModule
from pytorch_lightning import Trainer
from utils.config import Configuration

jsonconfig = Configuration.load_json('conf.json')
dm = LazyDataModule(data_dir_train = jsonconfig.output_train, 
                    data_dir_val = jsonconfig.output_validation, 
                    data_dir_test = jsonconfig.output_test, 
                    sources = jsonconfig.sources, 
                    mode = jsonconfig.mode, 
                    batch_size = jsonconfig.batch_size , 
                    num_workers = jsonconfig.num_workers)
model = SVSGAN()
trainer = Trainer(gpus=jsonconfig.gpus, max_epochs=jsonconfig.epoch, progress_bar_refresh_rate=20)
trainer.fit(model, dm)