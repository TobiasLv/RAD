import torch
from torch.optim.optimizer import Optimizer, required


class DLPF(Optimizer):
    r"""Implements dissipative leapfrog method.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        output_info (boolean, optional): whether to output the information of
          the training process (default: False)
    """

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, output_info=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, output_info=output_info)
        super(DLPF, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(DLPF, self).__setstate__(state)
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
            lr = group["lr"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        buf.mul_(momentum).add_(d_p, alpha=1 - momentum)
                    d_p = 1 / 2 * (momentum + 1) * buf
                else:
                    buf = d_p
                p.add_(d_p, alpha=-lr)

                exp_avg_norm_sq_total += torch.norm(buf) ** 2

        kinetic_energy = exp_avg_norm_sq_total * lr / (2 * (1 - momentum))

        if group["output_info"]:
            info_dict = {"kinetic_energy": kinetic_energy.item()}
            return loss, info_dict
        else:
            return loss
