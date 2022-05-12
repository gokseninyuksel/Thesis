# For data augmentation
from torchaudio.transforms import TimeMasking,FrequencyMasking 
from torch.utils.data import Dataset
import torch
import numpy as np
from utils.utils import load_numpy,soft_mask
import glob 

class LazyDataset(Dataset):
  def __init__(self,path, is_train = True, sources = ['vocals','drums'], mode = 'explicit'):
    '''
    Path to the pre_processed data. /content/drive/MyDrive/Thesis/MUSDB18_PreProcessed/test
    '''
    super(LazyDataset, self).__init__()
    mixture_files = glob.glob("{}/mixtures/*.npy".format(path))
    self.length = len(mixture_files) // 2
    del mixture_files
    self.path = path
    self.sources = sources
    self.mode = mode
    self.train = is_train
    if is_train:
      self.freq_mask = FrequencyMasking(freq_mask_param= 300)
      self.time_mask = TimeMasking(time_mask_param = 250,p = 1)
    if mode == 'SVSGAN':
      self.sources.append('background')
  def __len__(self):
    return self.length
  def __getitem__(self,index):
    # Load the spectrogram and phase for mixture
    seed = np.random.randint(low = 1, high = 1000000)
    mixture_spectrogram,phase_spectrogram = load_numpy(self.path, 'mixtures', index)
    mixture_spectrogram,phase_spectrogram =  torch.tensor(mixture_spectrogram, dtype = torch.float32),torch.tensor(phase_spectrogram, dtype = torch.float32)
    # # Load the mixture and phase for sources
    mixture_sources = torch.empty(len(self.sources), mixture_spectrogram.shape[0], mixture_spectrogram.shape[1])
    phase_sources = torch.empty_like(mixture_sources)
    for id in range(len(self.sources)):
      mixtures, sources = load_numpy(self.path, self.sources[id], index)
      mixture_sources[id] = torch.tensor(mixtures, dtype = torch.float32)
      phase_sources[id] = torch.tensor(sources, dtype = torch.float32)
    
    random = np.random.uniform(0,1)
    # When training, apply the data augmentation
    if self.train and random < 0.4:
      # 0.4 probability, apply the time mask to mixture and source spectrograms.
      torch.manual_seed(seed)
      mixture_spectrogram = self.time_mask(mixture_spectrogram[None,:,:])[0]
      for id_source in range(mixture_sources.shape[0]):
        torch.manual_seed(seed)
        spec = mixture_sources[id_source]
        mixture_sources[id_source] = self.time_mask(spec[None,:,:])[0]
      torch.manual_seed(seed)
      mixture_spectrogram = self.freq_mask(mixture_spectrogram)
      for id_source in range(mixture_sources.shape[0]):
        torch.manual_seed(seed)
        spec = mixture_sources[id_source]
        mixture_sources[id_source] = self.freq_mask(spec[None,:,:])[0]
    input = torch.concat((mixture_spectrogram[None,:,:],phase_spectrogram[None,:,:]), dim = 0)
    target = torch.concat((mixture_sources,phase_sources), dim = 0)
    return input,target