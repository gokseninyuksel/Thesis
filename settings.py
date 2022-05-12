from utils.config import Configuration
import torch
from torch.utils.tensorboard.writer import SummaryWriter

confjson = Configuration.load_json('conf.json')
global writer,sources_names,scaler,counter_train,counter_val
scaler = torch.cuda.amp.GradScaler(init_scale=10000,enabled = confjson.mixed_precision)
writer = SummaryWriter(confjson.writer_path,flush_secs = 5)
sources_names = confjson.sources_names
counter_train = 0 
counter_val = 0

