import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import tqdm.notebook as tq
import torch
import torch.nn as nn
from utils import utils

torch.manual_seed(0)
np.random.seed(0)

class DataLoader():
  def __init__(self,data,meta_data,stft_data):
    '''
    Data is expected to be a nussl class
    '''
    self.data = data
    self.sampling_rate = meta_data['sampling_rate']
    self.seconds = meta_data['seconds']
    self.nfft = stft_data['nfft']
    self.hop_size = stft_data['hop_size']
    self.dataPoints = 300032
    self.total_tracks = meta_data['total_tracks']
  def magnitude(self,verbose = 0,target_channel = 'mix', scale = 'linear', n_mels = 256):
    '''
    Return the magnitude of target channel in shape (total_tracks,1,1024,1024) as a torch tensor
    if scale = 'mel' shape is (total_tracks,1,n_mel,1024) 
    '''
    with torch.no_grad():
      magnitudes= torch.zeros((self.total_tracks, 1 ,  (self.nfft // 2 ) if scale == 'linear' else n_mels, (self.nfft // 2 )))
    for trackNr in tq.tqdm(range(self.total_tracks)) if verbose == 1 else range(self.total_tracks):

      #Get the channel data from the input argument. List composes the data and returns the mixture of the given channels.
      if target_channel == 'mix':
        channel = self.data[trackNr]['mix'].audio_data
      elif target_channel == 'vocals':
        channel = self.data[trackNr]['sources'][target_channel].audio_data
      elif isinstance(target_channel, list):
        channel = sum(self.data[trackNr]['sources'][ch].audio_data for ch in target_channel)
      #Turn stero into mono audio
      mono_audio = librosa.to_mono(channel)
      #Apply STFT
      stft_audio = librosa.stft(y = mono_audio,
                                n_fft = self.nfft,
                                hop_length = self.hop_size,
                                window = 'hann')
      #Cut off the last bin
      stft_audio = stft_audio[:-1, :-1]
      magnitude = utils().stft_magnitude(stft_audio)
      # Cut out the magnitudes that are smaller than 1, as they do not carry information
      magnitude[magnitude < 1] = 0
      with torch.no_grad():
        if scale == 'linear':
          magnitudes[trackNr] = magnitude
        elif scale == 'mel':
          mels =  librosa.feature.melspectrogram(sr=self.sampling_rate, 
                                                S=magnitude, 
                                                n_fft=self.nfft,
                                                hop_length=self.hop_size,
                                                n_mels = n_mels,
                                                power=1.0)
                                                
          magnitudes[trackNr] = torch.tensor(mels)
    return magnitudes
  def random_augment(self,prob,augmented):
    '''
    Randomly augment the vocal from augmented index by mixing the audio from the tracks.
    '''
    choices = ['drums', 'other']
    magnitudes = torch.empty((1,1,(self.nfft // 2 ), (self.nfft // 2 )))
    vocals = self.data[augmented]['sources']['vocals'].audio_data
    mono_vocal = librosa.to_mono(vocals)
    for trackNr in range(self.total_tracks):
      n = np.random.uniform(0,1)
      if n < prob:
        channel = np.random.choice(choices)
        channel = self.data[trackNr]['sources'][channel]
        mono_channel = librosa.to_mono(channel.audio_data)
        mono_mixed = mono_vocal + mono_channel
        stft_audio = librosa.stft(y = mono_mixed,
                                  n_fft = self.nfft,
                                  hop_length = self.hop_size,
                                  window = 'hann')
        stft_audio = stft_audio[:-1, :-1]
        stft_audio = utils().stft_magnitude(stft_audio)[None,None,:]
        magnitudes = torch.cat((magnitudes,stft_audio), dim = 0)
    return magnitudes[1:]