import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader
from model.UNet import UNet as Implicit
from model.Discriminator import Discriminator
from utils.config import Configuration
from model.multiSourceLoss import bce_loss_multiSource,l2_loss_multiSource
import settings 
from collections import OrderedDict
from utils.utils import weights_init_, seed_worker
from dataloader.LazyDataset import LazyDataset


class SVSGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.confjson = Configuration.load_json('conf.json')
        # Generator module for the GAN.
        self.generator = Implicit(baseline = self.confjson.baseline_generator,
                     in_channels = 1, 
                     nr_sources = len(settings.sources_names), 
                     dummy_conv_size =  self.confjson.dummy_generator,
                     mode = self.confjson.mode)
        # Discriminator module for the GAN
        self.discriminator =  Discriminator(in_channels = 1, baseline = self.confjson.baseline_discriminator,nr_sources = len(settings.sources_names))
        self.lossL2 = nn.MSELoss()
        self.loss_crossEntropy = nn.BCEWithLogitsLoss()
        self.generator.apply(weights_init_)
        self.discriminator.apply(weights_init_)
    def forward(self,X_batch):
        spec_inp = X_batch[:,:1,:,:]
        return spec_inp


    # ---------------------
    # TRAINING STEP
    # ---------------------
    def training_step(self,batch,batch_idx, optimizer_idx):
        X_batch,y_batch = batch 
        spec_inp = X_batch[:,:1,:,:]
        spec_target = y_batch[:,:settings.nr_sources,:,:]
        # Train Generator
        if optimizer_idx == 0:
            outputs = self.generator(spec_inp) if self.confjson.mode == 'implicit' else torch.multiply(self.generator(spec_inp),spec_inp)
            discriminator_fakes = self.discriminator(spec_inp, outputs)
            fakes = torch.ones(discriminator_fakes.shape)
            fakes = fakes.type_as(X_batch)
            # Calculate BCELoss for multi source
            bce_loss, _ = bce_loss_multiSource(discriminator_fakes,fakes, self.confjson.source_weights, self.loss_crossEntropy)
            # Calculate L2Loss for multi source
            l2_loss, _ = l2_loss_multiSource(outputs,spec_target,self.confjson.source_weights,self.lossL2)
            gan_loss = bce_loss + self.confjson.alpha * l2_loss
            self.log('training_gan_loss', gan_loss)
            self.log('training_bce_loss', bce_loss)
            self.log('training_l2_loss', l2_loss)
            return gan_loss
        if optimizer_idx == 1:
            fakes = self.generator(spec_inp) if self.confjson.mode == 'implicit' else torch.multiply(self.generator(spec_inp),spec_inp)
            discriminator_fakes = self.discriminator(spec_inp, fakes.detach())
            discriminator_reals = self.discriminator(spec_inp, spec_target)
            # Calculate BCELoss Fake for multi source
            zeros = torch.zeros(discriminator_fakes.shape,requires_grad = True)
            zeros = zeros.type_as(X_batch)
            ones = torch.ones(discriminator_fakes.shape, requires_grad = True)
            ones = ones.type_as(X_batch)
            fake_loss, _ = bce_loss_multiSource(discriminator_fakes, zeros,self.confjson.source_weights, self.loss_crossEntropy)
            real_loss, _ = bce_loss_multiSource(discriminator_reals, ones,self.confjson.source_weights, self.loss_crossEntropy)
            gan_loss = (fake_loss + real_loss) / 2
            self.log('training_gan_loss_discriminator', gan_loss)
            return gan_loss
    
    # ---------------------
    #  VALIDATION STEP
    # ---------------------
    def validation_step(self, batch, batch_idx):
        X_batch,y_batch = batch 
        spec_inp = X_batch[:,:1,:,:]
        spec_target = y_batch[:,:settings.nr_sources,:,:]

        # Validation step for the generator
        outputs = self.generator(spec_inp) if self.confjson.mode == 'implicit' else torch.multiply(self.generator(spec_inp),spec_inp)
        discriminator_fakes = self.discriminator(spec_inp, outputs)
        fakes = torch.ones(discriminator_fakes.shape)
        fakes = fakes.type_as(X_batch)
        # Calculate BCELoss for multi source
        bce_loss, _ = bce_loss_multiSource(discriminator_fakes,fakes, self.confjson.source_weights, self.loss_crossEntropy)
        # Calculate L2Loss for multi source
        l2_loss,source_losses_l2 = l2_loss_multiSource(outputs,spec_target,self.confjson.source_weights,self.lossL2)
        gan_loss = bce_loss + self.confjson.alpha * l2_loss
        self.log('validation_gan_loss', gan_loss)
        self.log('validation_l2_loss', l2_loss)
        self.log('validation_bce_loss', bce_loss)
        # Validation step for the discrimiantor
        fakes = self.generator(spec_inp) if self.confjson.mode == 'implicit' else torch.multiply(self.generator(spec_inp),spec_inp)
        discriminator_fakes = self.discriminator(spec_inp, fakes.detach())
        discriminator_reals = self.discriminator(spec_inp, spec_target)
        # Calculate BCELoss Fake for multi source
        zeros = torch.zeros(discriminator_fakes.shape)
        zeros = zeros.type_as(X_batch)
        ones = torch.ones(discriminator_fakes.shape)
        ones = ones.type_as(X_batch)
        fake_loss, _ = bce_loss_multiSource(discriminator_fakes, zeros,self.confjson.source_weights, self.loss_crossEntropy)
        real_loss, _ = bce_loss_multiSource(discriminator_reals, ones,self.confjson.source_weights, self.loss_crossEntropy)
        gan_loss = (fake_loss + real_loss) / 2
        self.log('training_gan_loss_discriminator', gan_loss)
    # ---------------------
    #  TEST STEP
    # ---------------------
    def test_step(self,batch,batch_idx):
        X_batch,y_batch = batch 
        spec_inp = X_batch[:,:1,:,:]
        spec_target = y_batch[:,:settings.nr_sources,:,:]
        outputs = self.generator(spec_inp) if self.confjson.mode == 'implicit' else torch.multiply(self.generator(spec_inp),spec_inp)
        discriminator_fakes = self.discriminator(spec_inp, outputs)
        fakes = torch.ones(discriminator_fakes.shape)
        fakes = fakes.type_as(X_batch)
        # Calculate BCELoss for multi source
        bce_loss, _ = bce_loss_multiSource(discriminator_fakes,fakes, self.confjson.source_weights, self.loss_crossEntropy)
        # Calculate L2Loss for multi source
        l2_loss,source_losses_l2 = l2_loss_multiSource(outputs,spec_target,self.confjson.source_weights,self.lossL2)
        gan_loss = bce_loss + self.confjson.alpha * l2_loss
        self.log('test_gan_loss', gan_loss)
        self.log('test_l2_loss', l2_loss)
        self.log('test_bce_loss', bce_loss)     


    # ---------------------
    #  OPTIMIZER CONFIGURATION
    # ---------------------  
    def configure_optimizers(self):
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(),lr = self.confjson.discriminator_lr)
        optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr = self.confjson.generator_lr, weight_decay = 0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_generator,
                                                                mode = 'min',
                                                                factor = 0.5,
                                                                patience  = 10,
                                                                threshold=0.001,
                                                                verbose = True)
        lr_schedulers = {"scheduler": scheduler, "monitor": "validation_gan_loss"}
        return [optimizer_generator,optimizer_discriminator], lr_schedulers
    def train_dataloader(self):
        train_data = LazyDataset(path = self.confjson.output_train,
                                 is_train = True,
                                 sources = settings.sources_names,
                                 mode = self.confjson.mode)
        g = torch.Generator()
        g.manual_seed(0)
        return  DataLoader(train_data,
                        batch_size = self.confjson.batch_size, 
                        shuffle = True, 
                        num_workers = self.confjson.num_workers, 
                        worker_init_fn=seed_worker,
                        generator= g,
                        pin_memory = True,
                        )
    def val_dataloader(self):
        val_data = LazyDataset(path = self.confjson.output_validation,
                                 is_train = False,
                                 sources = settings.sources_names,
                                 mode = self.confjson.mode)
        g = torch.Generator()
        g.manual_seed(0)
        return  DataLoader(val_data,
                        batch_size = self.confjson.batch_size, 
                        shuffle = False, 
                        num_workers = self.confjson.num_workers, 
                        worker_init_fn=seed_worker,
                        generator= g,
                        pin_memory = True,
                        )

    def test_dataloader(self):
        test_data = LazyDataset(path = self.confjson.output_test,
                                 is_train = False,
                                 sources = settings.sources_names,
                                 mode = self.confjson.mode)
        g = torch.Generator()
        g.manual_seed(0)
        return  DataLoader(test_data,
                        batch_size = self.confjson.batch_size, 
                        shuffle = False, 
                        num_workers = self.confjson.num_workers, 
                        worker_init_fn=seed_worker,
                        generator= g,
                        pin_memory = True,
                        )  
