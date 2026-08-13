import torch
from torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    r"""Implements Adam algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant (default: False)
        output_info (boolean, optional): whether to output the information of
          the training process (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False, output_info=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            output_info=output_info,
        )
        super(Adam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
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

        kinetic_energy = 0
        exp_avg_norm = 0
        exp_avg_sq_norm = 0
        efficient_lr_norm = 0
        step_size_norm = 0
        grad_norm = 0
        grad_sq_norm = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                weight_decay = group["weight_decay"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if weight_decay != 0:
                    grad = grad.add(p, alpha=weight_decay)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                if group["amsgrad"]:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(state["max_exp_avg_sq"], exp_avg_sq, out=state["max_exp_avg_sq"])
                    # Use the max. for normalizing running avg. of gradient
                    denom = 1 / (torch.sqrt(state["max_exp_avg_sq"] / bias_correction2) + group["eps"])
                else:
                    denom = 1 / (torch.sqrt(exp_avg_sq / bias_correction2) + group["eps"])

                p.addcmul_(exp_avg, denom, value=-group["lr"] / bias_correction1)

                if group["output_info"]:
                    kinetic_energy += group["lr"] * torch.sum(torch.sqrt((exp_avg**2) / ((1 - beta1) ** 2) + 1))
                    exp_avg_norm += torch.sum(exp_avg**2)
                    exp_avg_sq_norm += torch.sum(exp_avg_sq**2)
                    efficient_lr = denom * group["lr"] / bias_correction1
                    efficient_lr_norm += torch.sum(efficient_lr**2)
                    step_size = exp_avg * efficient_lr
                    step_size_norm += torch.sum(step_size**2)
                    grad_norm += torch.sum(grad**2)
                    grad_sq_norm += torch.sum(grad**4)

        if group["output_info"]:
            exp_avg_norm = torch.sqrt(exp_avg_norm)
            exp_avg_sq_norm = torch.sqrt(exp_avg_sq_norm)
            efficient_lr_norm = torch.sqrt(efficient_lr_norm)
            step_size_norm = torch.sqrt(step_size_norm)
            grad_norm = torch.sqrt(grad_norm)
            grad_sq_norm = torch.sqrt(grad_sq_norm)

            info_dict = {
                "kinetic_energy": kinetic_energy.item(),
                "exp_avg_norm": exp_avg_norm.item(),
                "exp_avg_sq_norm": exp_avg_sq_norm.item(),
                "efficient_lr_norm": efficient_lr_norm.item(),
                "step_size_norm": step_size_norm.item(),
                "grad_norm": grad_norm.item(),
                "grad_sq_norm": grad_sq_norm.item(),
            }
            return loss, info_dict
        else:
            return loss
