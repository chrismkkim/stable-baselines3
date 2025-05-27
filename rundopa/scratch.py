import matplotlib.pyplot as plt
import numpy as np

eps = 1e-8
f = lambda x: x/(x**2 + eps)

x = np.logspace(start=-8, stop=0, num=100, base=10)
y = f(x)

plt.figure()
plt.plot(x, y)
plt.xscale('log')
plt.title('f = x/(x^2 + eps), eps = 1e-5')
plt.tight_layout()
plt.savefig('figure/logscale_plot.png', dpi=300)
# plt.show()






grads = th.cat([
    p.grad.view(-1)
    for p in self.policy.parameters()
    if p.grad is not None
])

frac = grads[th.abs(grads)<0.5].shape[0] / grads.shape[0]

import matplotlib.pyplot as plt
plt.figure()
# plt.hist(grads, bins=100, histtype='step')
plt.hist(grads, bins=100, range=(-10,10), histtype='step')
plt.yscale('log')
plt.tight_layout()
plt.show()





grad_dict1 = {}
# View gradients
for name, param in self.policy.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
        grad_dict1[name] = param.grad.norm()
        
        
grad_dict2 = {}
# View gradients
for name, param in self.policy.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
        grad_dict2[name] = param.grad.norm()
    
    
for key in grad_dict1.keys():
    print(grad_dict1[key] == grad_dict2[key])
    