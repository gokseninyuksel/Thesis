import torch

class Dataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, X, y):
        'Initialization'
        self.labels = X
        self.targets = y

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.labels)

  def __getitem__(self, index):
        'Generates one sample of data'
        return self.labels[index], self.targets[index]
  def sample(self,device):
        with torch.no_grad():
          prob = torch.tensor([1/self.labels.shape[0]])
          prob = prob.repeat(self.labels.shape[0])
          prob = prob.float()
          prob.to(device)
          index = torch.distributions.categorical.Categorical(probs= prob).sample()
        return self.labels[index].to(device), self.targets[index].to(device)