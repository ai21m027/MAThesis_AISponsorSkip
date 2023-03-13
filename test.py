import torch.nn as nn
import torch


calc = torch.Tensor([[1,0],[1,0],[0,1],[0,1],[1,0]])
true = torch.Tensor([0,0,1,1,0])
true = true.type(torch.long)
#true = torch.empty(5,dtype=torch.long).random_(2)

loss =nn.CrossEntropyLoss()
print(loss(calc, true))

if idx == 0:
    targets.append(0)
elif idx == len(sentences) - 1:
    # 1 if last sentence is sponsor, 0 if not to differentiate between sponsor and beginning of next video in batch
    targets.append(entry[idx][2])
else:
    if entry[idx - 1][2] != entry[idx][2]:
        targets.append(1)
    else:
        targets.append(0)