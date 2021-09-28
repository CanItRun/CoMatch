from torch import nn
import torch

attn = nn.MultiheadAttention(128, 4)
x = torch.rand(3, 32, 128)
xn, w = attn.forward(x, x, x)
print(xn.shape)
print(w)
