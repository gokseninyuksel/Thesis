
import torch 
from model.generator_train import train_generator,test_generator
from model.discriminator_train import train_discriminator,test_discriminator
from torch import nn
import tqdm as tqdm 
import gc 
from utils.config import Configuration
import settings 
from utils.utils import weights_init_, set_momentum
import random
import numpy as np
from utils.utils import compute_sdr

confjson = Configuration.load_json('conf.json')
def train_GAN_step(discriminator,generator,
                   optimizer_discriminator, optimizer_generator,
                   loss_crossEntropy,loss_L2,
                   train_iter,val_iter,
                   device, source_weights, epoch, scheduler, mixed_precision = True, 
                   alpha = 100,nr_sources = 1):
  global writer,sources_names,counter_train,counter_val
  discriminator.train()
  generator.train()
  acc_loss_generator_train_L2, acc_loss_generator_train_BCE, acc_loss_generator_train_GAN = 0,0,0
  acc_loss_discriminator_fake, acc_loss_discriminator_real, acc_loss_discriminator_total = 0,0,0
  acc_loss_generator_test_L2, acc_loss_generator_test_BCE, acc_loss_generator_test_GAN = 0,0,0
  acc_loss_discriminator_test_fake, acc_loss_discriminator_test_real, acc_loss_discriminator_test_total = 0,0,0
  acc_sdr_train, acc_sdr_test = torch.zeros(len((train_iter.dataset))),torch.zeros(len((val_iter.dataset)))
  source_losses_train_acc = 0
  source_losses_test_acc = 0
  len_train = len((train_iter.dataset))
  len_val = len((val_iter.dataset))
  # Train the Generator
  for X_batch,y_batch in train_iter:
    X_batch, y_batch = X_batch.to(torch.float32), y_batch.to(torch.float32)
    X_batch,y_batch = X_batch.to(device), y_batch.to(device)
    loss_generator_train_L2, loss_generator_train_BCE, loss_generator_train_GAN,source_losses_train,sdr_train=  train_generator(
                    X_batch,y_batch,
                    discriminator,generator,
                    optimizer_generator,
                    loss_crossEntropy,loss_L2,
                    device = device, mixed_precision = mixed_precision, source_weights = source_weights,alpha = alpha, nr_sources = nr_sources)
    source_losses_train_acc = source_losses_train + source_losses_train_acc
    acc_loss_generator_train_L2 += loss_generator_train_L2.detach()
    acc_loss_generator_train_BCE += loss_generator_train_BCE.detach()
    acc_loss_generator_train_GAN += loss_generator_train_GAN.detach()
    sdr_train[settings.counter_train] = sdr_train
    settings.counter_train += 1
    settings.writer.add_scalar("Training Generator GAN Loss Step",loss_generator_train_GAN/X_batch.shape[0], settings.counter_train) 
    # Train the Discriminator
    loss_discriminator_fake, loss_discriminator_real, loss_discriminator_total = train_discriminator( 
                    X_batch,y_batch,
                    discriminator,generator,
                    optimizer_discriminator,
                    loss_crossEntropy,
                    device = device, 
                    mixed_precision=mixed_precision,
                    source_weights = source_weights, nr_sources = nr_sources)
    
    acc_loss_discriminator_fake += loss_discriminator_fake.detach()
    acc_loss_discriminator_real += loss_discriminator_real.detach()
    acc_loss_discriminator_total  += loss_discriminator_total.detach()
    # if counter_train % 20 == 0:
    #   plot_random_sample(generator,'Training' ,  X_batch,y_batch,counter_train,'log')
  discriminator.eval()
  generator.eval()
  # Test the Generator
  for X_batch,y_batch in val_iter:
    X_batch, y_batch = X_batch.to(torch.float32), y_batch.to(torch.float32)
    X_batch,y_batch = X_batch.to(device).detach(), y_batch.to(device).detach()
    loss_generator_test_L2, loss_generator_test_BCE, loss_generator_test_GAN,source_losses_test, sdr_test =  test_generator(
                    X_batch,y_batch, 
                    discriminator,generator,
                    optimizer_generator,
                    loss_crossEntropy,loss_L2,
                   device = device, source_weights = source_weights,alpha = alpha, nr_sources = nr_sources)
    acc_loss_generator_test_L2 += loss_generator_test_L2
    acc_loss_generator_test_BCE += loss_generator_test_BCE
    acc_loss_generator_test_GAN += loss_generator_test_GAN
    source_losses_test_acc = source_losses_test+ source_losses_test_acc
    acc_sdr_test += sdr_test
    settings.counter_val += 1
    settings.writer.add_scalar("Validation Generator GAN Loss Step",loss_generator_test_GAN/X_batch.shape[0], settings.counter_val) 
    # Test the Discriminator
    loss_discriminator_test_fake, loss_discriminator_test_real, loss_discriminator_test_total = test_discriminator(
                    X_batch,y_batch,
                    discriminator,generator,
                    optimizer_discriminator,
                    loss_crossEntropy,
                   device = device, source_weights = source_weights, nr_sources = nr_sources)
    # if counter_val % 10 == 0:
      # plot_random_sample(generator,'Validation' , X_batch,y_batch,counter_val,'log')
    acc_loss_discriminator_test_fake += loss_discriminator_test_fake
    acc_loss_discriminator_test_real += loss_discriminator_test_real
    acc_loss_discriminator_test_total += loss_discriminator_test_total
    sdr_test[settings.counter_val] = sdr_test
  scheduler.step(acc_loss_generator_test_L2 / len_val)
  for source_index in range(nr_sources):
    settings.writer.add_scalar(f"Training Generator {settings.sources_names[source_index]} L2 Loss", source_losses_train_acc[source_index]/ len_train, epoch)
  settings.writer.add_scalar("Training Generator L2 Loss", acc_loss_generator_train_L2 / len_train, epoch)
  settings.writer.add_scalar("Training Generator BCE Loss", acc_loss_generator_train_BCE / len_train, epoch)
  settings.writer.add_scalar("Training Generator GAN Loss", acc_loss_generator_train_GAN / len_train, epoch)
  settings.writer.add_scalar("Training Discriminator Fake Loss", acc_loss_discriminator_fake / len_train, epoch)
  settings.writer.add_scalar("Training Discriminator Real Loss", acc_loss_discriminator_real / len_train, epoch)
  settings.writer.add_scalar("Training Discriminator Total Loss", acc_loss_discriminator_total / len_train, epoch) 
  settings.writer.add_scalar("Trainind Generator SDR",  torch.median(sdr_train) , epoch)
  for source_index in range(nr_sources):
    settings.writer.add_scalar(f"Validation Generator {settings.sources_names[source_index]} L2 Loss", source_losses_test_acc[source_index]/ len_val, epoch)
  settings.writer.add_scalar("Validation Generator L2 Loss", acc_loss_generator_test_L2 / len_val, epoch)
  settings.writer.add_scalar("Validation Generator BCE Loss", acc_loss_generator_test_BCE / len_val, epoch)
  settings.writer.add_scalar("Validation Generator GAN Loss", acc_loss_generator_test_GAN / len_val, epoch)
  settings.writer.add_scalar("Validation Discriminator Fake Loss", acc_loss_discriminator_test_fake / len_val, epoch)
  settings.writer.add_scalar("Validation Discriminator Real Loss", acc_loss_discriminator_test_real / len_val, epoch)
  settings.writer.add_scalar("Validation Discriminator Total Loss", acc_loss_discriminator_test_total / len_val, epoch) 
  settings.writer.add_scalar("Validation Generator SDR",  acc_sdr_test / len_val , epoch)


def train_GAN(discriminator,generator, 
              train_iter,test_iter,
              nr_epochs, device,source_weights, lr_generator = 0.001, lr_discriminator = 0.001,
              alpha = 100,nr_sources = 1, mixed_precision = True):
  '''
  Train the generator adverserial network.
  disc -> Discriminator network
  train_iter -> Training set iterator for the data
  test_iter -> Test set iterator for the data
  nr_epochs -> Total number of epochs to train the discriminator
  device -> GPU if it is available otherwise cpu
  lr -> Learning rate for the discriminator network
  '''
  optimizer_discriminator = torch.optim.Adam(discriminator.parameters(),lr = lr_discriminator)
  optimizer_generator = torch.optim.Adam(generator.parameters(), lr = lr_generator, weight_decay = 0.01)
  loss_crossEntropy = nn.BCEWithLogitsLoss()
  loss_MultiSource = nn.MSELoss()
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_generator,
                                                         mode = 'min',
                                                         factor = 0.5,
                                                         patience  = 10,
                                                         threshold=0.001,
                                                         verbose = True)

  for epoch in tqdm.tqdm(range(nr_epochs)):
    train_GAN_step(discriminator,generator,
                   optimizer_discriminator, optimizer_generator,
                   loss_crossEntropy,loss_MultiSource,
                   train_iter,test_iter,
                   device,
                   source_weights,epoch,
                   mixed_precision = mixed_precision, 
                   alpha = alpha, nr_sources = nr_sources,scheduler = scheduler)
    torch.save(generator.state_dict(), confjson.generator_weight.format(epoch + 1))
    torch.save(discriminator.state_dict(),  confjson.discriminator_weight.format(epoch + 1))
    gc.collect()
    torch.cuda.empty_cache()