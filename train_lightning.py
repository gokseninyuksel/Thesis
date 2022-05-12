import json
from trainer import SVSGAN
from dataloader.LazyDataset import LazyDataset
from pytorch_lightning import Trainer
from utils.config import Configuration
from utils.utils import seed_worker
from torch.utils.data import DataLoader
import settings 
settings.init()

jsonconfig = Configuration.load_json('conf.json')
output_validation =  jsonconfig.output_validation
output_train =  jsonconfig.output_train
output_test =  jsonconfig.output_test
train_data = LazyDataset(path = output_train, 
                        is_train = True ,
                        sources = settings.
                        sources_names, mode = jsonconfig.mode)
val_data = LazyDataset(path = output_validation,    
                       is_train = False, 
                       sources = settings.sources_names, 
                       mode = jsonconfig.mode)
test_data = LazyDataset(path = output_test,
                       is_train = False, 
                       sources = settings.sources_names, 
                       mode = jsonconfig.mode)

train_iter = DataLoader(train_data,
                        batch_size = jsonconfig.batch_size, 
                        shuffle = True, 
                        num_workers = jsonconfig.num_workers, 
                        worker_init_fn=seed_worker,
                        generator=g,
                        pin_memory = True,
)
val_iter = DataLoader(val_data,
                      batch_size = 24,
                      shuffle = True, 
                      num_workers = 4,
                      worker_init_fn=seed_worker,
                      generator=g,
                      pin_memory = True)
test_iter = DataLoader(test_data,
                      batch_size = 24,
                      shuffle = False, 
                      num_workers = 4,
                      worker_init_fn=seed_worker,
                      generator=g,
                      pin_memory = True)
model = SVSGAN()
trainer = Trainer(gpus=jsonconfig.gpus, 
                  precision = 16, 
                  max_epochs=jsonconfig.epoch, progress_bar_refresh_rate=20)
trainer.fit(model,train_iter,val_iter,test_iter)