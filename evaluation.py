import settings
import torch
import librosa 
from utils.utils import combine_magnitude_phase, compute_sdr
import numpy as np 
from mir_eval.separation import bss_eval_sources
import sys
from model.UNet import UNet as Net
from utils.config import Configuration
from torch.utils.data import DataLoader
from utils.utils import seed_worker
from dataloader.LazyDataset import LazyDataset

confjson = Configuration.load_json('conf.json')
def my_collate(batch):
    batch = list(filter(lambda x: not torch.any(torch.isnan(x[1])) , batch))
    return torch.utils.data.dataloader.default_collate(batch)

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
  settings.init()
  g = torch.Generator()
  g.manual_seed(0)
  device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
  generator = Net(baseline = confjson.baseline_generator,
                     in_channels = 1, 
                     nr_sources = settings.nr_sources, 
                     dummy_conv_size =  confjson.dummy_generator,
                     mode = confjson.mode).to(device)
  generator.load_state_dict(torch.load(sys.argv[1]))
  val_data = LazyDataset(path = confjson.output_validation, is_train = False, sources = settings.sources_names, mode = confjson.mode)
  test_data = LazyDataset(path = confjson.output_test, is_train = False, sources = settings.sources_names, mode = confjson.mode)
  val_iter = DataLoader(val_data,
                      batch_size = 24,
                      shuffle = False, 
                      num_workers = 4,
                      worker_init_fn=seed_worker,
                      generator=g,
                      pin_memory = False,
                      collate_fn=my_collate)
  test_iter = DataLoader(test_data,
                      batch_size = 24,
                      shuffle = False, 
                      num_workers = 4,
                      worker_init_fn=seed_worker,
                      generator=g,
                      pin_memory = False,
                      collate_fn=my_collate)
  print("Successfuly initiated the weights")
  print("Computing SDRs for validation and test")
  sdrs_validation = compute_eval_scores(generator,val_iter,device, mode = sys.argv[2])
  print('Computed Sdrs for validation')
  sdrs_test = compute_eval_scores(generator,test_iter, device, mode = sys.argv[2])
  print('Computed Sdrs for test')
  for id,(val,test) in enumerate(zip(sdrs_validation,sdrs_test)):
    print("For source {}, \n sdr validation is: {} \n sdr test is: {}".format(confjson.source_names[id],np.median(val),np.median(test)))