# RAD Optimizer Family

This repository is the official PyTorch implementation of the RAD optimizer family, including `RAD`, `RADAR`, and the comparison optimizers used in the associated research.

The PyPI distribution is `pytorch-rad`, the Python package is `rad`, and all optimizers are available from `rad.optim`.

## Installation

Install from PyPI:

```bash
pip install pytorch-rad
```

Alternatively, install directly from GitHub:

```bash
pip install git+https://github.com/TobiasLv/RAD.git
```

The package requires Python 3.6 or later, PyTorch, and NumPy.

## RAD-family optimizers

### RAD

RAD (Relativistic Adaptive Gradient Descent) is a physics-inspired optimizer designed to improve the stability of deep reinforcement learning. It models neural-network training as a conformal Hamiltonian system and uses relativistic kinetic energy to impose a finite update speed, reducing the influence of abnormal gradients while providing parameter-wise adaptive learning rates. Across five RL algorithms and twelve environments, RAD improved training stability and performance over a broad set of optimizer baselines.

```python
from rad.optim import RAD

optimizer = RAD(model.parameters(), lr=1e-3)
```

Set `zeta=None` to use the schedule controlled by `max_iter`:

```python
optimizer = RAD(
    model.parameters(),
    lr=1e-3,
    zeta=None,
    max_iter=total_steps,
)
```

### RADAR

RADAR (Relativistic Adaptive Gradient Descent with Accelerated Residual) is developed from the ADMM-Inspired Momentum (AIM) framework, which interprets momentum as a multiplier-like correction driven by a splitting residual. It combines relativistic adaptive geometry, decoupled residual correction, and second-order momentum filtering to improve both update directions and momentum estimates. Experiments in vision learning, language modeling, and reinforcement learning show consistent improvements over strong adaptive-optimizer baselines.

```python
from rad.optim import RADAR

optimizer = RADAR(model.parameters(), lr=1e-3)
```

RADAR defaults:

| Parameter | Default | Purpose |
| --- | ---: | --- |
| `lr` | `1e-3` | Learning rate |
| `betas` | `(0.9, 0.999)` | First- and second-moment coefficients |
| `gamma` | `0.01` | Gradient-residual correction coefficient |
| `l` | `None` | Residual correction step size |
| `delta` | `1` | Adaptive-preconditioner scaling coefficient |
| `zeta` | `1e-16` | Numerical-stability coefficient |
| `weight_decay` | `0` | Weight-decay coefficient |
| `decoupled_weight_decay` | `True` | Use decoupled decay when `weight_decay > 0` |

When `l=None`, each parameter group initializes

$$
l = 0.01 \times \mathrm{lr}_{\mathrm{initial}}.
$$

With the default `lr=1e-3`, this gives `l=1e-5`. The value of `l` then remains fixed when a learning-rate scheduler changes `lr`. You can override it explicitly:

```python
optimizer = RADAR(model.parameters(), lr=1e-3, l=5e-6)
```

Weight decay is disabled by default. To enable it, set a positive coefficient; decay is decoupled by default:

```python
optimizer = RADAR(model.parameters(), weight_decay=1e-2)
```

Set `decoupled_weight_decay=False` only when coupled weight decay is intended.

RADAR works with standard PyTorch learning-rate schedulers:

```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=1000,
)
```

## Available optimizers

All optimizers can be imported from `rad.optim`:

```python
from rad.optim import (
    AdaBayes,
    Adam,
    AdamW,
    DLPF,
    KFAdam,
    NAdam,
    NAG,
    RAD,
    RADAR,
    RGD,
    SGD,
    SWATS,
)
```

## Papers and citation

If you find the RAD optimizer family useful, please consider giving this repository a star ⭐ and citing the corresponding papers in your work.

### RAD

RAD was introduced in **Conformal Symplectic Optimization for Stable Reinforcement Learning**.

<a href="https://ieeexplore.ieee.org/document/10792938">
    <img src="https://github.com/user-attachments/assets/80e4d671-51d7-46a3-b27a-08b5e08a3051" alt="IEEE Xplore" width="92">
</a>
<a href="https://arxiv.org/abs/2412.02291">
    <img src="https://img.shields.io/badge/arXiv-PDF-red?style=flat&logo=arXiv&logoColor=white" alt="arXiv PDF">
</a>

```bibtex
@ARTICLE{lyu2025conformal,
  author={Lyu, Yao and Zhang, Xiangteng and Li, Shengbo Eben and Duan, Jingliang and Tao, Letian and Xu, Qing and He, Lei and Li, Keqiang},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  title={Conformal Symplectic Optimization for Stable Reinforcement Learning},
  year={2025},
  volume={36},
  number={6},
  pages={11049-11063}
}
```

### RADAR

RADAR is introduced in **Momentum as Residual-Driven Multiplier Correction for Deep Learning Optimization**.

ArXiv paper and citation coming soon.

## Repository organization

```text
RAD/
├── rad/
│   └── optim/                  # One implementation module per optimizer
├── tests/                      # Import and optimizer behavior tests
└── Supplementary materials/    # Original RAD paper materials
```

The supplementary PDF contains materials associated with the RAD paper.

## Development

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
```

Build the source distribution and wheel with:

```bash
python -m pip install build
python -m build
```

## License

This project is licensed under the [MIT License](./LICENSE).
