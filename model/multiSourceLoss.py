
import torch 

def l2_loss_multiSource(outputs,spec_target,source_weights, loss_L2):
  '''
  For each source, calculate the l2 loss and sum them.
  total_loss = vocal_loss + background_loss 
  '''
  with torch.no_grad():
    source_losses = torch.empty((len(source_weights),))
  l2_loss = 0
  for source_index in range(len(source_weights)):
      source_loss = loss_L2(outputs[:,source_index,:,:], spec_target[:,source_index,:,:])
      l2_loss += source_weights[source_index] * source_loss
      source_losses[source_index] = loss_L2(outputs[:,source_index,:,:], spec_target[:,source_index,:,:]).detach()
  return l2_loss,source_losses

def bce_loss_multiSource(predicted,target,source_weights, loss_bce):
  '''
  For each source, calculate the l2 loss and sum them.
  total_loss = vocal_loss + background_loss 
  '''
  with torch.no_grad():
    source_losses = torch.empty((len(source_weights),))
  bce_loss = 0
  for source_index in range(len(source_weights)):
      source_loss = loss_bce(predicted[:,source_index,:,:], target[:,source_index,:,:])
      bce_loss += source_weights[source_index] * source_loss
      source_losses[source_index] = loss_bce(predicted[:,source_index,:,:], target[:,source_index,:,:]).detach()
  return bce_loss,source_losses