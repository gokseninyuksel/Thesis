
import torch
from multiSourceLoss import bce_loss_multiSource, l2_loss_multiSource


def train_generator(X_batch,y_batch,discriminator,generator,
                    optimizer_generator,
                    loss_crossEntropy,loss_L2,
                    source_weights,device, alpha = 100, nr_sources = 1, mixed_precision = True):
    '''
    Train the generator on X_batch and y_batch.
    X_batch: The input batch for the generator
    y_batch: The target batch for the generator
    discriminator: Discriminator module from the discriminator.py
    generator: Generator module from the UNet.py
    optimizer_generator: Adam optimizer from torch.optim.Adam
    loss_crossEntropy: Loss object from torch.nn.CrossEntropy.
    loss_L2: Loss object from torch.nn.MSELoss()
    source_weights: Specify the weighted multi source.
    '''
    global scaler
    # Input spectrogram and targets spectrogram
    spec_inp = X_batch[:,:1,:,:]
    spec_target = y_batch[:,:nr_sources,:,:]
    # Set the optimizer generator zero grad
    optimizer_generator.zero_grad(set_to_none=True)
    with torch.cuda.amp.autocast(enabled = mixed_precision):
      outputs = generator(spec_inp)
      discriminator_fakes = discriminator(spec_inp, outputs)
      # Calculate BCELoss for multi source
      bce_loss, _ = bce_loss_multiSource(discriminator_fakes, torch.ones(discriminator_fakes.shape).to(device),source_weights, loss_crossEntropy)
      # Calculate L2Loss for multi source
      l2_loss,source_losses_l2 = l2_loss_multiSource(outputs,spec_target,source_weights,loss_L2)
      gan_loss = bce_loss + alpha * l2_loss
    # Back-propagate the loss
    scaler.scale(gan_loss).backward()
    # Update weights
    scaler.step(optimizer_generator)
    loss_generator_L2 = l2_loss.detach()
    loss_generator_BCE = bce_loss.detach()
    loss_generator_GAN = gan_loss.detach()
    scaler.update()


    return loss_generator_L2, loss_generator_BCE, loss_generator_GAN, source_losses_l2

def test_generator(X_batch,y_batch,discriminator,generator,
                    optimizer_generator,
                    loss_crossEntropy,loss_L2,
                    device, source_weights, alpha = 100,nr_sources = 1):
  
    # Input spectrogram and target spectrogram
    spec_inp = X_batch[:,:1,:,:]
    spec_target = y_batch[:,:nr_sources,:,:]
    # Compute the spectrogram from the U-Net, add the extra channel to get (1,1,x,x) dimensions, also get the probabilities from discriminator
    with torch.no_grad():
        outputs = generator(spec_inp)
        discriminator_fakes = discriminator(spec_inp, outputs)
        # Calculate BCELoss for multi source
        with torch.cuda.amp.autocast(enabled = False):
          bce_loss, _= bce_loss_multiSource(discriminator_fakes, torch.ones(discriminator_fakes.shape).to(device),source_weights, loss_crossEntropy)
          # Calculate L2Loss for multi source
          l2_loss,source_losses_l2 = l2_loss_multiSource(outputs,spec_target,source_weights,loss_L2)
          # Calculate the GAN LOSS
          gan_loss = bce_loss + alpha * l2_loss
          loss_generator_L2 = l2_loss.detach()
          loss_generator_BCE = bce_loss.detach()
          loss_generator_GAN = gan_loss.detach()
    return loss_generator_L2, loss_generator_BCE, loss_generator_GAN,source_losses_l2