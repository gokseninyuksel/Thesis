from model.UNet import UNet as Implicit
from model.Discriminator import Discriminator
from dataloader.LazyDataset import LazyDataset 
from model.train_GAN import train_GAN
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import numpy as np 
import random
from utils.config import Configuration
from utils.utils import weights_init_, set_momentum,seed_worker
import settings 
from torch import nn 
import atexit
from evaluation import compute_eval_scores

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

def exit_handler():
  sds_val = compute_eval_scores(generator, val_iter,device)
  sds_t = compute_eval_scores(generator,test_iter, device)
  for val,test in zip(sds_val,sds_t):
    print(np.median(val), np.median(test))
atexit.register(exit_handler)



def my_collate(batch):
    batch = list(filter(lambda x: not torch.any(torch.isnan(x[1])) , batch))
    return torch.utils.data.dataloader.default_collate(batch)

  
settings.init()
confjson = Configuration.load_json('conf.json')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
source_weights = confjson.source_weights
nr_sources = len(settings.sources_names)
output_validation =  confjson.output_validation
output_train =  confjson.output_train
output_test = confjson.output_test
generator = Implicit(baseline = confjson.baseline_generator,
                     in_channels = 1, 
                     nr_sources = nr_sources, 
                     dummy_conv_size =  confjson.dummy_generator,
                     mode = confjson.mode)
print("U_NET with mode {}".format(generator.mode))
print("Using Mixed Precision:{}".format(confjson.mixed_precision))
print("Training on sources: {}".format(confjson.source_names))
discriminator = Discriminator(in_channels = 1, baseline = confjson.baseline_discriminator,nr_sources = nr_sources)
generator.to(device)
discriminator.to(device)
train_data = LazyDataset(path = output_train, is_train = True ,sources = settings.sources_names, mode = confjson.mode)
val_data = LazyDataset(path = output_validation, is_train = False, sources = settings.sources_names, mode = confjson.mode)
test_data = LazyDataset(path = output_test, is_train = False, sources = settings.sources_names, mode = confjson.mode)
g = torch.Generator()
g.manual_seed(0)
train_iter = DataLoader(train_data,
                        batch_size = confjson.batch_size, 
                        shuffle = True, 
                        num_workers = confjson.num_workers, 
                        worker_init_fn=seed_worker,
                        generator=g,
                        pin_memory = True,
                        collate_fn=my_collate if confjson.filter_nan else None
)
val_iter = DataLoader(val_data,
                      batch_size = 24,
                      shuffle = False, 
                      num_workers = 4,
                      worker_init_fn=seed_worker,
                      generator=g,
                      pin_memory = True,
                      collate_fn=my_collate if confjson.filter_nan else None)
test_iter = DataLoader(test_data,
                      batch_size = 24,
                      shuffle = False, 
                      num_workers = 4,
                      worker_init_fn=seed_worker,
                      generator=g,
                      pin_memory =True,
                      collate_fn=my_collate if confjson.filter_nan else None)
print('Created the train and validation itertors with size {}, {}'.format(len(train_data), len(val_data)))
generator.apply(weights_init_)
discriminator.apply(weights_init_)
set_momentum(generator,0.1)
set_momentum(discriminator,0.1)
train_GAN(discriminator,
            generator, 
            train_iter,
            val_iter,
            confjson.epoch, 
            device,
            mixed_precision = confjson.mixed_precision,
            source_weights = source_weights,lr_generator = confjson.generator_lr, lr_discriminator = confjson.discriminator_lr, nr_sources = nr_sources,alpha = confjson.alpha
            )