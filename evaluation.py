import settings
import torch
import librosa 
from utils.utils import combine_magnitude_phase, compute_sdr
import numpy as np 
from mir_eval.separation import bss_eval_sources
import argparse
 
parser = argparse.ArgumentParser(description="Evaluating the pre-trained model",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g", "--generator", action="store_true", help="generator weights")
parser.add_argument("-d", "--discriminator", action="store_true", help="discriminator weights")
parser.add_argument("-m", "--method", help="New or old method for calculating SDR")
args = parser.parse_args()

def compute_eval_scores(net,data,device, mode = 'new'):
    net.eval()
    sdr = [[] for a in range (len(settings.sources_names))]
    nr_sources = len(settings.sources_names)
    for X_sampled,y_sampled in data:
      with torch.no_grad():
        X_sampled,y_sampled = X_sampled.to(device),y_sampled.to(device)
        # Convert sample to float32, otherwise it is not compatible with the network.
        X_sampled,y_sampled = X_sampled.to(torch.float32), y_sampled.to(torch.float32)
        X_sampled,y_sampled = X_sampled.detach(), y_sampled.detach()
        target_spec = y_sampled[:,:nr_sources,:,:]
        input_spec = X_sampled[:,:1,:,:]
        target_phase = y_sampled[:,nr_sources:,:,:]
        input_phase = X_sampled[:,1:,:,:]
        y_hat = net(input_spec).detach()

      for index in range(X_sampled.shape[0]):
        for id, sources in enumerate(settings.sources_names):
          stft = combine_magnitude_phase(y_hat[index,id].cpu().numpy(),input_phase[index,0].detach().cpu().numpy())
          pred_audio = librosa.istft(stft, hop_length = 300)
          x_audio = combine_magnitude_phase(input_spec[index,0,].cpu().numpy(),input_phase[index,0].detach().cpu().numpy())
          y_audio = combine_magnitude_phase(target_spec[index, id].cpu().numpy(),target_phase[index, id].detach().cpu().numpy())
          x_audio = librosa.istft(x_audio, hop_length = 300)
          y_audio = librosa.istft(y_audio, hop_length = 300)
          if not np.all(y_audio == 0) and not np.all(pred_audio == 0): 
            sdr_eval= compute_sdr(y_audio,pred_audio) if mode == 'new' else bss_eval_sources(y_audio,pred_audio)
            sdr[id].append(sdr_eval)
    return sdr

if __name__ == "__main__":
  print(args)