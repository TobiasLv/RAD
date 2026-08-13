import math

import torch
from torch.optim.optimizer import Optimizer


class AdaBayes(Optimizer):
    r"""Implements AdaBayes Fixed Point algorithm (https://arxiv.org/abs/1807.07540; NeurIPS 2020)

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): analogous to Adam learning rate (default: 1e-3).  In the high-data limit, converges to Adam(W) with this learning rate.
        lr_sgd (float, optional): analogous to SGD learning rate (default: 1e-1).  In the low-data limit, converges to SGD with this learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): Decoupled weight decay
        batch_size: The batch size.  Necessary for correct normalization.

    Note:
        Assumes that the loss is the mean loss, averaged over the minibatch
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        lr_sgd=1e-1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=5e-5,
        batch_size=1,
        output_info=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_sgd:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(
            lr=lr,
            lr_sgd=lr_sgd,
            lr_ratio=lr_sgd / lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            weight_decay_ratio=weight_decay / lr,
            batch_size=batch_size,
            output_info=output_info,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

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
                grad = p.grad.data
                grad.mul_(group["batch_size"])

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data)
                    ## Posterior uncertainty
                    state["s2post"] = torch.ones_like(p.data).fill_(group["lr_sgd"] / group["batch_size"])

                #### Adam!
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                #### Steady state Bayesian filtering
                lr_sgd = group["lr"] * group["lr_ratio"] / group["batch_size"]

                eta = group["lr"]
                eta2 = eta**2
                sigma2 = lr_sgd
                var_decay = math.exp(-eta2 / sigma2)
                var_add = sigma2 * (1 - math.exp(-eta2 / sigma2))

                s2post = state["s2post"]
                # s2post(t-1) -> s2prior(t)
                s2post.mul_(var_decay).add_(var_add)
                ###s2prior(t) -> s2post(t)
                s2post.reciprocal_().addcmul_(grad, grad).reciprocal_()

                p.data.addcmul_(exp_avg, s2post, value=-1 / bias_correction1)

                if 0 != group["weight_decay"]:
                    p.data.mul_(1 - group["weight_decay_ratio"] * group["lr"])

                if group["output_info"]:
                    exp_avg_norm += torch.sum(exp_avg**2)
                    exp_avg_sq_norm += torch.sum(1 / s2post**4)
                    efficient_lr_norm += torch.sum((s2post / bias_correction1) ** 2)
                    step_size_norm += torch.sum((exp_avg / bias_correction1 * s2post) ** 2)
                    grad_norm += torch.sum(grad**2)
                    grad_sq_norm += torch.sum(grad**4)

        if group["output_info"]:
            info_dict = {
                "kinetic_energy": 0,
                "exp_avg_norm": torch.sqrt(exp_avg_norm).item(),
                "exp_avg_sq_norm": torch.sqrt(exp_avg_sq_norm).item(),
                "efficient_lr_norm": torch.sqrt(efficient_lr_norm).item(),
                "step_size_norm": torch.sqrt(step_size_norm).item(),
                "grad_norm": torch.sqrt(grad_norm).item(),
                "grad_sq_norm": torch.sqrt(grad_sq_norm).item(),
            }
            return loss, info_dict
        else:
            return loss
