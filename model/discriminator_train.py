import torch
from multiSourceLoss import bce_loss_multiSource, l2_loss_multiSource

def train_discriminator(X_batch,y_batch,discriminator,generator,
                   optimizer_discriminator,
                   loss_crossEntropy,
                   device,source_weights, nr_sources = 1,mixed_precision = True):

    global scaler
    # Input spectrogram and target spectrogram
    spec_inp = X_batch[:,:1,:,:]
    reals = y_batch[:,:nr_sources,:,:]
    optimizer_discriminator.zero_grad(set_to_none=True)
    # Compute the spectrogram from the U-Net, add the extra channel to get (1,1,x,x) dimensions
    with torch.cuda.amp.autocast(enabled = mixed_precision):
      fakes = generator(spec_inp) if generator.mode == 'implicit' else torch.multiply(generator(spec_inp),spec_inp)
      discriminator_fakes = discriminator(spec_inp, fakes.detach())
      discriminator_reals = discriminator(spec_inp, reals)
      # Calculate BCELoss Fake for multi source
      fake_loss, _ = bce_loss_multiSource(discriminator_fakes, torch.zeros(discriminator_fakes.shape).to(device),source_weights, loss_crossEntropy)
      real_loss, _ = bce_loss_multiSource(discriminator_reals, torch.ones(discriminator_fakes.shape).to(device),source_weights, loss_crossEntropy)
      gan_loss = (fake_loss + real_loss) / 2
    # Back-propagate the loss
    scaler.scale(gan_loss).backward()
    # Update weights
    scaler.step(optimizer_discriminator)
    scaler.update()
    loss_discriminator_fake = fake_loss.detach()
    loss_discriminator_real = real_loss.detach()
    loss_discriminator_total = gan_loss.detach()
    return loss_discriminator_fake, loss_discriminator_real,  loss_discriminator_total
def test_discriminator(X_batch,y_batch,discriminator,generator,
                   optimizer_discriminator,
                   loss_crossEntropy,
                   device,source_weights,nr_sources = 1):
    # Input spectrogram and target spectrogram
    spec_inp = X_batch[:,:1,:,:]
    reals = y_batch[:,:nr_sources,:,:]
    with torch.no_grad():
      # Compute the spectrogram from the U-Net, add the extra channel to get (1,1,x,x) dimensions
      fakes =  generator(spec_inp) if generator.mode == 'implicit' else torch.multiply(generator(spec_inp),spec_inp)
      discriminator_fakes = discriminator(spec_inp, fakes.detach())
      discriminator_reals = discriminator(spec_inp, reals.detach())
      # Calculate the L2Loss
      with torch.cuda.amp.autocast(enabled = False):
        fake_loss, _ = bce_loss_multiSource(discriminator_fakes, torch.zeros(discriminator_fakes.shape).to(device),source_weights, loss_crossEntropy)
        real_loss, _ = bce_loss_multiSource(discriminator_reals, torch.ones(discriminator_fakes.shape).to(device),source_weights, loss_crossEntropy)
        gan_loss = (fake_loss + real_loss) / 2
        loss_discriminator_fake = fake_loss.detach()
        loss_discriminator_real = real_loss.detach()
        loss_discriminator_total = gan_loss.detach()

    return loss_discriminator_fake, loss_discriminator_real,  loss_discriminator_total 