import torch
import torch.nn as nn
import librosa
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import tqdm.notebook as tq
import random 
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)
def spectrogram(stft_array,sampling_rate, scale = 'linear', hop_length = 256, ax = None):
    """
    Function for displaying the spectrogram. It is expected that if torch tensor is passed, shape is (1,n,m) else (n,m)

    stft_array -> Array containing magnitudes, or complex number.
    sampling_rate -> Sampling Rate for the signal.
    scale -> Display mode of the spectrogram.
    hop_length -> Stft hop length.
    """ 
    if stft_array.shape[0] == 1:
      stft_array = torch.squeeze(stft_array)
      stft_array = stft_array.detach().cpu().numpy()
    if ax:
      img = librosa.display.specshow(stft_array,sr = sampling_rate, x_axis = 'time', hop_length = hop_length, y_axis = scale, ax = ax,cmap ='RdBu_r')
      plt.colorbar(img, ax = ax)
    else:
      img = librosa.display.specshow(stft_array,sr = sampling_rate, x_axis = 'time',hop_length = hop_length, y_axis = scale,cmap ='RdBu_r')
      plt.colorbar()
def plot_wave(wave,sampling_rate,ax = None):
    '''
    Display the time series wave
    '''
    if ax:
      librosa.display.waveshow(y = wave, x_axis = 'time', sr = sampling_rate, ax = ax)
    else:
      librosa.display.waveshow(y = wave, sr = sampling_rate, x_axis = 'time')
def play_audio(audio,sampling_rate):
    '''
    Play the specified audio with specified sampling_rate
    '''
    return Audio(audio, rate = sampling_rate)
def stft_magnitude(stft):
    '''
    Calculate the stft_magnitude
    '''
    return torch.tensor(np.abs(stft))
def stft_angle(stft):
    '''
    Calculate the stft_angle
    '''
    return torch.tensor(np.angle(stft))
def soft_mask(source,mixture):
    '''
    Calculate the soft mask from source and mixture
    returns the soft mask.
    '''
    return source / (mixture + 0.0001)
def combine_magnitude_phase(magnitudes, phases):
    '''
    Combine the phase information from the stft array to magnitude information.
    This method produces the stft array with imaginary numbers, which can then be
    used in ISTFT

    returns the magnitude and phase combined array
    '''

    return magnitudes * np.exp(1.j * phases)
def load_numpy(path,source,index):
    '''
    Load the saved numpy file from the path. Path is expected to have /spectrogram_{}.npy and /phase_{}.npy files.
    Return the loaded spectrogram and phase
    '''
    spectrogram = np.load(path + f'/{source}/spectrogram_{index}.npy')
    phase = np.load(path + f'/{source}/phase_{index}.npy')
    return spectrogram,phase
def compute_sdr(ref,est):
    '''
    Compute the sdr from reference and estimated source.
    returns the sdr in DB.
    '''
    ratio = np.sum(ref**2) / np.sum((ref-est)**2)
    return 10*np.log10(ratio + 1e-10)
def compute_sdr_(ref,est):
    '''
    Compute the sdr from reference and estimated source. Pytorch
    returns the sdr in DB.
    '''
    ratio = torch.sum(ref**2) / torch.sum((ref-est)**2)
    return 10*torch.log10(ratio + 1e-10)
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def weights_init_(m):
        # for every convolution layer in a model..
        if isinstance(m, nn.Conv2d):
            # Init with xavier weights
            torch.nn.init.xavier_normal_(m.weight)
def set_momentum(net,value):
    for a in net.children():
            for b in a.children():
                if isinstance(b, nn.BatchNorm2d):
                    b.momentum = value