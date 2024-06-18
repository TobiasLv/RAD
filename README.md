# RAD (Relativistic Adaptive Gradient Descent)
## Description
Training deep reinforcement learning (RL) agents necessitates overcoming the highly unstable nonconvex stochastic optimization inherent in the trial-and-error mechanism. To tackle this challenge, we propose a physics-inspired optimization algorithm called relativistic adaptive gradient descent (RAD), which enhances long-term training stability. By conceptualizing neural network (NN) training as the evolution of a conformal Hamiltonian system, we present a universal framework for transferring long-term stability from conformal symplectic integrators to iterative NN updating rules, where the choice of kinetic energy governs the dynamical properties of resulting optimization algorithms. By utilizing relativistic kinetic energy, RAD incorporates principles from special relativity and limits parameter updates below a finite speed, effectively mitigating abnormal gradient influences. Additionally, RAD models NN optimization as the evolution of a multi-particle system where each trainable parameter acts as an independent particle with an individual adaptive learning rate. We prove RAD's sublinear convergence under general nonconvex settings, where smaller gradient variance and larger batch sizes contribute to tighter convergence. Notably, RAD degrades to the well-known adaptive moment estimation (ADAM) algorithm when its speed coefficient is chosen as one and symplectic factor as a small fixed positive value. Experimental results on MuJoCo and Atari benchmarks show that RAD achieves state-of-the-art performance compared to cutting-edge optimizers, emphasizing its potential for stabilizing RL training.

## Requirement
1. Linux is preferred.
2. Python 3.6 or greater.
3. Pytorch installed.

## Quick Start
Installing the package is straightforward with pip directly from this git repository or from pypi with either of the following commands.

```bash
pip install git+https://github.com/TobiasLv/RAD
```

```bash
pip install pytorch-rad
```

All optimizers have been implemented in the Python file "optimizers.py", including RAD, Adam, SGD (equaling HB when momentum is not 0), DLPF, NAG, RGD, NAdam, SWATS, AdamW. After installing the package, you can import any of these optimizers and use them in your code as any other `torch.optim.Optimizer`

```python
from rad.optim import RAD, Adam, SGD, DLPF, RGD, NAG, NAdam, SWATS, AdamW

# Example usage:
# max_iter is optional, but recommended for fast convergence,
# usually as the maximum number of network updates.
rad_optim = RAD(net.parameters(), lr=0.001, max_iter=max_iter)
adam_optim = Adam(net.parameters(), lr=0.001)
hb_optim = SGD(net.parameters(), lr=0.001, momentum=0.9)
dlpf_optim = DLPF(net.parameters(), lr=0.001, momentum=0.9)
rgd_optim = RGD(net.parameters(), lr=0.001, momentum=0.9)
nag_optim = NAG(net.parameters(), lr=0.001, momentum=0.9)
sgd_optim = SGD(net.parameters(), lr=0.001, momentum=0)
nadam_optim = NAdam(net.parameters(), lr=0.001)
swats_optim = SWATS(net.parameters(), lr=0.001)
adamw_optim = AdamW(net.parameters(), lr=0.001)
```

## Supplementary materials
Any user can find the supplementary in the "Supplementary materials" folder.