import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import tqdm.notebook as tq
import torch
import torch.nn as nn
from DataLoader import DataLoader

torch.manual_seed(0)
np.random.seed(0)

class DataLoaderFactory():
  """
  Factory method for data loaders.
  """

  def __init__(self,train,test):
    #Training samples
    self.train = train
    #Test samples
    self.test = test
    #Data specifications for the train
    self.meta_data_train =  {
    'sampling_rate' : 44100,
    'seconds': 7,
    'total_tracks': len(train.items)
        }
    #Data specifications for the test
    self.meta_data_test = {
        'sampling_rate' : 44100,
        'seconds': 7,
        'total_tracks': len(test.items)
    }
    #Stft parameters to calculate stft of the signal.
    self.stft_params = {
        'nfft' : 2048,
        'hop_size' : 293
    }

  def create(self,data):
    """ 
    Return the data loader class depending on the type that has passed
    """
    return DataLoader(self.test,self.meta_data_test,self.stft_params) if data == "test" else DataLoader(self.train,self.meta_data_train,self.stft_params)

  def magnitude(self,data,scale = 'linear',channel = "mix"):
    """
    Returns the magnitude of specified channel from the specified portion of data
    """
    loader = self.create(data)
    magnitude= loader.magnitude(verbose = 1, scale = scale,target_channel = channel)
    return magnitude
  def augment(self, prob = 1):
    """
    Augment the training data by making mixtures with various song components and vocals 
    prob -> Probability of augmenting
    """
    loader = self.create("train")
    print("----Importing the magnitude of training vocals-----")
    train_vocals = loader.magnitude(verbose = 1, target_channel = 'vocals')
    print("----Importing the magnitude of training mixes-----")
    train_mixtures = loader.magnitude(verbose = 1, target_channel = 'mix')
    print("----Starting the augmentation process-----")
    for track_nr in tq.tqdm(range(loader.total_tracks), position = 0, ):
      magnitude_vocal = train_vocals[track_nr]
      random_augmented = loader.random_augment(prob,track_nr)
      magnitude_vocal = magnitude_vocal.repeat(random_augmented.shape[0],1, 1, 1)
      train_vocals = torch.cat((train_vocals,magnitude_vocal))
      train_mixtures = torch.cat((train_mixtures, random_augmented))
    return train_vocals, train_mixtures 