import torch
from Phi.model import Phi


x = torch.randint(0, 256, (1, 1024)).cuda()

Phi(x) # (1, 1024, 20000)
