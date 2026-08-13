import torch
from torch.optim.optimizer import Optimizer, required


class RGD(Optimizer):
    r"""Implements relativistic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0 for standard
            SGD with lr = lr/2)
        delta (float, optional): strength of normalization (default: 0 for a
            2-order CM method)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        order (int, optional): precision of the approximation to the relativistic
            Hamiltonian system
        output_info (boolean, optional): whether to output the information of
          the training process (default: False)
    """

    def __init__(self, params, lr=required, momentum=0, delta=0, weight_decay=0, order=1, output_info=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if delta < 0.0:
            raise ValueError("Invalid delta value: {}".format(delta))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if order not in [1, 2]:
            raise ValueError("Invalid order: {}".format(order))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            delta=delta,
            weight_decay=weight_decay,
            order=order,
            output_info=output_info,
        )
        super(RGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("output_info", False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        exp_avg_norm_sq_total = 0
        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            delta = group["delta"]
            lr = group["lr"]
            order = group["order"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                else:
                    buf = param_state["momentum_buffer"]
                    buf.mul_(momentum).add_(d_p, alpha=1 - momentum)
                d_p = buf
                exp_avg_norm_sq = torch.norm(buf) ** 2
                lr_k = 1 / torch.sqrt((delta**2) * exp_avg_norm_sq + 1)
                if order == 1:
                    lr_k *= 2
                elif order == 2:
                    lr_k += 1 / torch.sqrt((delta**2) * exp_avg_norm_sq + 1 / (momentum**2))
                lr_k *= lr / 2
                p.add_(d_p, alpha=-lr_k)

                exp_avg_norm_sq_total += exp_avg_norm_sq

        if delta != 0:
            kinetic_energy = lr / delta * torch.sqrt(exp_avg_norm_sq_total / ((1 - momentum) ** 2) + 1 / (delta**2))
        else:
            kinetic_energy = exp_avg_norm_sq_total * lr / (2 * (1 - momentum))

        if group["output_info"]:
            info_dict = {"kinetic_energy": kinetic_energy.item()}
            return loss, info_dict
        else:
            return loss
