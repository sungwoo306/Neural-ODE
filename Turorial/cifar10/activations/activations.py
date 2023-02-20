import torch

e = torch.exp(torch.tensor(1.0))
pi = torch.pi

# simply define a kelu function
def molu(input):
    return input * torch.tanh(torch.exp(input))
