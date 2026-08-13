import math
from typing import List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class AdamW(Optimizer):
    r"""Implements AdamW algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (boolean, optional): whether to use the AMSGrad variant (default: False)
        output_info (boolean, optional): whether to output the information of
          the training process (default: False)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        amsgrad=False,
        output_info=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            output_info=output_info,
        )
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("output_info", False)

    def adamw(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        max_exp_avg_sqs: List[Tensor],
        state_steps: List[int],
        *,
        amsgrad: bool,
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        eps: float,
        output_info: bool,
    ):
        r"""Functional API that performs AdamW algorithm computation.

        See :class:`~torch.optim.AdamW` for details.
        """
        kinetic_energy = 0
        exp_avg_norm = 0
        exp_avg_sq_norm = 0
        efficient_lr_norm = 0
        step_size_norm = 0
        grad_norm = 0
        grad_sq_norm = 0
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]

            # Perform stepweight decay
            param.mul_(1 - lr * weight_decay)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1

            param.addcdiv_(exp_avg, denom, value=-step_size)

            if output_info:
                kinetic_energy += lr * torch.sum(torch.sqrt((exp_avg**2) / ((1 - beta1) ** 2) + 1))
                exp_avg_norm += torch.sum(exp_avg**2)
                exp_avg_sq_norm += torch.sum(exp_avg_sq**2)
                efficient_lr = step_size / denom
                efficient_lr_norm += torch.sum(efficient_lr**2)
                step_size_norm += torch.sum((exp_avg * efficient_lr) ** 2)
                grad_norm += torch.sum(grad**2)
                grad_sq_norm += torch.sum(grad**4)
        if output_info:
            info_dict = {
                "kinetic_energy": kinetic_energy.item(),
                "exp_avg_norm": torch.sqrt(exp_avg_norm).item(),
                "exp_avg_sq_norm": torch.sqrt(exp_avg_sq_norm).item(),
                "efficient_lr_norm": torch.sqrt(efficient_lr_norm).item(),
                "step_size_norm": torch.sqrt(step_size_norm).item(),
                "grad_norm": torch.sqrt(grad_norm).item(),
                "grad_sq_norm": torch.sqrt(grad_sq_norm).item(),
            }
        else:
            info_dict = {}
        return info_dict

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            state_sums = []
            max_exp_avg_sqs = []
            state_steps = []
            amsgrad = group["amsgrad"]
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                grads.append(p.grad)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if amsgrad:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])

                # update the steps for each param group update
                state["step"] += 1
                # record the step after step update
                state_steps.append(state["step"])

            info_dict = self.adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                output_info=group["output_info"],
            )

        if group["output_info"]:
            return loss, info_dict
        else:
            return loss
