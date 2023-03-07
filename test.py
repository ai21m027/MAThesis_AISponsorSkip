import torch.nn as nn
import torch


calc = torch.Tensor([[1,0],[1,0],[0,1],[0,1],[1,0]])
true = torch.Tensor([0,0,1,1,0])
true = true.type(torch.long)
#true = torch.empty(5,dtype=torch.long).random_(2)

loss =nn.CrossEntropyLoss()
print(loss(calc, true))