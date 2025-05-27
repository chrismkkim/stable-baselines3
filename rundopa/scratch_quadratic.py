import numpy as np
import torch as th
import torch.nn as nn
from torch.nn import functional as F

import matplotlib.pyplot as plt


model = nn.Sequential(
    nn.Linear(4,64),
    nn.Tanh(),
    nn.Linear(64,64),
    nn.Tanh(),
    nn.Linear(64,64),
    nn.Tanh(),
    nn.Linear(64,1)    
)

optimizer = th.optim.Adam(model.parameters(), lr=1e-4)

nepoch = 50000
ndata = 100
m0 = 0
m1 = 10
save_loss = th.zeros(nepoch)
# mi = (m1-m0) * time_step / total_timesteps + m0
for epoch in range(nepoch):
    with th.no_grad():
        rewards = th.ones(ndata)
        old_values = th.normal(mean=th.tensor(0), std=th.tensor(0.3), size=(ndata,))
        next_values = th.normal(mean=th.tensor(0), std=th.tensor(0.3), size=(ndata,))
        next_dones = (th.rand(ndata) < 0.5).float()
        # next_dones = th.rand(ndata)
        advantages = rewards + (th.tensor(1)-next_dones) * next_values - old_values
        
        # Convert to tensor
        rewards_tensor     = th.as_tensor(rewards).view(-1,1)
        values_tensor      = th.as_tensor(old_values).view(-1,1)
        next_values_tensor = th.as_tensor(next_values).view(-1,1)
        next_dones_tensor  = th.as_tensor(next_dones).view(-1,1)
            
        input = th.cat([rewards_tensor, next_values_tensor, values_tensor, next_dones_tensor], dim=1)
                                
    dopa = model(input)

    loss = F.mse_loss(advantages, dopa.flatten())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    save_loss[epoch] = loss.detach().item()


save_loss = th.sqrt(save_loss)

plt.figure()
plt.plot(th.log10(save_loss), marker='.', linestyle='')
plt.axhline(-2, color='b', linestyle='--')
plt.tight_layout()
plt.show()


x=1