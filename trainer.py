import pytorch_lightning as pl
import torch
from torch import nn
from model.UNet import UNet as Implicit
from model.Discriminator import Discriminator
from utils.config import Configuration
from model.multiSourceLoss import bce_loss_multiSource,l2_loss_multiSource
import settings 
from collections import OrderedDict
from utils.utils import weights_init_


settings.init()

class SVSGAN(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.confjson = Configuration.load_json('../conf.json')
        self.generator = Implicit(baseline = self.confjson.baseline_generator,
                     in_channels = 1, 
                     nr_sources = settings.nr_sources, 
                     dummy_conv_size =  self.confjson.dummy_generator,
                     mode = self.confjson.mode)
        self.discriminator =  Discriminator(in_channels = 1, baseline = self.confjson.baseline_discriminator,nr_sources = settings.nr_sources)
        self.lossL2 = nn.MSELoss()
        self.loss_crossEntropy = nn.BCEWithLogitsLoss()
        self.generator.apply(weights_init_)
        self.discriminator.apply(weights_init_)
    def forward(self,X_batch):
        spec_inp = X_batch[:,:1,:,:]
        return spec_inp
    def training_step(self,batch,batch_idx,optimizer_idx):
        print("Traning Step")
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
            bce_loss, _ = bce_loss_multiSource(discriminator_fakes,fakes, self.source_weights, self.loss_crossEntropy)
            # Calculate L2Loss for multi source
            l2_loss,source_losses_l2 = l2_loss_multiSource(outputs,spec_target,self.confjson.source_weights,self.lossL2)
            gan_loss = bce_loss + self.confjson.alpha * l2_loss
            tqdm_dict = {"g_loss": gan_loss,
                        "bce_loss" : bce_loss,
                        "l2_loss": l2_loss}
            output = OrderedDict({"loss": gan_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output
        if optimizer_idx == 1:
            fakes = self.generator(spec_inp) if self.confjson.mode == 'implicit' else torch.multiply(self.generator(spec_inp),spec_inp)
            discriminator_fakes = self.discriminator(spec_inp, fakes.detach())
            discriminator_reals = self.discriminator(spec_inp, reals)
            # Calculate BCELoss Fake for multi source
            zeros = torch.zeros(discriminator_fakes.shape)
            zeros = zeros.type_as(X_batch)
            ones = torch.ones(discriminator_fakes.shape)
            ones = ones.type_as(X_batch)
            fake_loss, _ = bce_loss_multiSource(discriminator_fakes, zeros,self.confjson.source_weights, self.loss_crossEntropy)
            real_loss, _ = bce_loss_multiSource(discriminator_reals, ones,self.confjson.source_weights, self.loss_crossEntropy)
            gan_loss = (fake_loss + real_loss) / 2
            tqdm_dict = {"d_loss": gan_loss}
            output = OrderedDict({"loss": gan_loss, "progress_bar": tqdm_dict, "log": tqdm_dict})
            return output
    def validation_step(self, batch, batch_idx):
        print("Validation Step")
        X_batch,y_batch = batch 
        spec_inp = X_batch[:,:1,:,:]
        spec_target = y_batch[:,:settings.nr_sources,:,:]
        outputs = self.generator(spec_inp) if self.confjson.mode == 'implicit' else torch.multiply(self.generator(spec_inp),spec_inp)
        discriminator_fakes = self.discriminator(spec_inp, outputs)
        fakes = torch.ones(discriminator_fakes.shape)
        fakes = fakes.type_as(X_batch)
        # Calculate BCELoss for multi source
        bce_loss, _ = bce_loss_multiSource(discriminator_fakes,fakes, self.source_weights, self.loss_crossEntropy)
        # Calculate L2Loss for multi source
        l2_loss,source_losses_l2 = l2_loss_multiSource(outputs,spec_target,self.confjson.source_weights,self.lossL2)
        gan_loss = bce_loss + self.confjson.alpha * l2_loss
        tqdm_dict = {"g_loss": gan_loss,
                        "bce_loss" : bce_loss,
                        "l2_loss": l2_loss}
        self.log('validation g_loss', gan_loss)
        self.log('validation l2_loss', l2_loss)
        self.log('validation bce_loss', bce_loss)
    def test_step(self,batch,batch_idx):
        X_batch,y_batch = batch 
        spec_inp = X_batch[:,:1,:,:]
        spec_target = y_batch[:,:settings.nr_sources,:,:]
        outputs = self.generator(spec_inp) if self.confjson.mode == 'implicit' else torch.multiply(self.generator(spec_inp),spec_inp)
        discriminator_fakes = self.discriminator(spec_inp, outputs)
        fakes = torch.ones(discriminator_fakes.shape)
        fakes = fakes.type_as(X_batch)
        # Calculate BCELoss for multi source
        bce_loss, _ = bce_loss_multiSource(discriminator_fakes,fakes, self.source_weights, self.loss_crossEntropy)
        # Calculate L2Loss for multi source
        l2_loss,source_losses_l2 = l2_loss_multiSource(outputs,spec_target,self.confjson.source_weights,self.lossL2)
        gan_loss = bce_loss + self.confjson.alpha * l2_loss
        tqdm_dict = {"g_loss": gan_loss,
                        "bce_loss" : bce_loss,
                        "l2_loss": l2_loss}
        self.log('test g_loss', gan_loss)
        self.log('test l2_loss', l2_loss)
        self.log('testbce_loss', bce_loss)       
    def configure_optimizers(self):
        optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(),lr = self.confjson.discriminator_lr)
        optimizer_generator = torch.optim.Adam(self.generator.parameters(), lr = self.confjson.generator_lr, weight_decay = 0.01)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_generator,
                                                                mode = 'min',
                                                                factor = 0.5,
                                                                patience  = 10,
                                                                threshold=0.001,
                                                                verbose = True)
        return [optimizer_generator, optimizer_discriminator], [scheduler]


    
