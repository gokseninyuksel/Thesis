from model.UNet import UNet as Implicit
from model.Discriminator import Discriminator
from dataloader.LazyDataset import LazyDataset 
from model.train_GAN import train_GAN
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from utils import seed_worker
import numpy as np 
import random
from utils.config import Configuration


confjson = Configuration.load_json('conf.json')
global writer,sources_names,scaler
global counter_train,counter_val

mixed_precision = True
counter_train,counter_val = 0,0
scaler = torch.cuda.amp.GradScaler(init_scale=10000,enabled = mixed_precision)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

writer = SummaryWriter(confjson.writer_path,flush_secs = 5)
sources_names = confjson.sources_names
source_weights = confjson.sources_weights
nr_sources = len(sources_names)
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

output_validation =  confjson.output_validation
output_train =  confjson.output_train
generator = Implicit(baseline = confjson.baseline_generator, in_channels = 1, nr_sources = nr_sources, dummy_conv_size =  confjson.dummy_generator).to(device)
discriminator = Discriminator(in_channels = 1, baseline = confjson.baseline_discriminator,nr_sources = nr_sources).to(device)
train_data = LazyDataset(path = output_train, is_train = True ,sources = sources_names, mode = 'implicit')
val_data = LazyDataset(path = output_validation, is_train = False, sources = sources_names, mode = 'implicit')
g = torch.Generator()
g.manual_seed(0)
train_iter = DataLoader(train_data,
                        batch_size = confjson.batch_size, 
                        shuffle = True, 
                        num_workers = confjson.num_workers, 
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

train_GAN(discriminator,
            generator, 
            train_iter,
            val_iter,
            confjson.epoch, 
            device,
            mixed_precision = mixed_precision
            source_weights = source_weights,lr_generator = confjson.generator_lr, lr_discriminator = confjson.discriminator_lr, nr_sources = nr_sources,alpha = confjson.alpha
            )