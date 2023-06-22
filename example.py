import torch
from PHI import phi2


x = torch.randint(0, 256, (1, 1024)).cuda()

phi2(x) # (1, 1024, 20000)
