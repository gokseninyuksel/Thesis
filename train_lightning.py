from lightning.trainer import SVSGAN
from pytorch_lightning import Trainer
from utils.config import Configuration
import settings 
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
import torch 

torch.manual_seed(0)
seed_everything(0)

settings.init()
jsonconfig = Configuration.load_json('conf.json')
model = SVSGAN()
checkpoint_path = ".."
checkpoint = ModelCheckpoint(
            checkpoint_path, monitor='validation_gan_loss', mode='min', save_top_k=1, verbose=1, save_last=True)
logger = TensorBoardLogger(
        save_dir='./logger',
        version=1,
        name='lightning_logs'
    )
trainer = Trainer(gpus=jsonconfig.gpus, 
                  precision = 16, 
                  checkpoint_callback=checkpoint,
                  max_epochs=jsonconfig.epoch, 
                  progress_bar_refresh_rate=20,
                  logger=logger
                   )
trainer.fit(model)