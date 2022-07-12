

import torch

u = torch.FloatTensor([[5,6.1],[7,8]])
w = torch.FloatTensor([[0.5,0.5],[0.5,0.5]])

res = torch.FloatTensor([[3,6],[7,6]])

case1 = torch.where(res<(u-w), res+w, res)
mask1 = torch.where((res - case1)!=0, 1, 0)

case3 = torch.where(res>(u+w), res-w, res)
mask3 = torch.where((res - case3)!=0, 1, 0)

mask2 = torch.ones_like(mask1) - mask1 -mask3

reso = case1*mask1 + u*mask2 + case3*mask3

i = 1
































