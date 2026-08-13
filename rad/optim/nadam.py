from typing import List

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class NAdam(Optimizer):
    r"""Implements NAdam algorithm.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum_decay (float, optional): momentum momentum_decay (default: 4e-3)
        output_info (boolean, optional): whether to output the information of
          the training process (default: False)
    """

    def __init__(
        self,
        params,
        lr=2e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        momentum_decay=4e-3,
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
        if not 0.0 <= momentum_decay:
            raise ValueError("Invalid momentum_decay value: {}".format(momentum_decay))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
            output_info=output_info,
        )
        super(NAdam, self).__init__(params, defaults)

    def nadam(
        self,
        params: List[Tensor],
        grads: List[Tensor],
        exp_avgs: List[Tensor],
        exp_avg_sqs: List[Tensor],
        mu_products: List[float],
        state_steps: List[int],
        beta1: float,
        beta2: float,
        lr: float,
        weight_decay: float,
        momentum_decay: float,
        eps: float,
        output_info: bool,
    ):
        r"""Functional API that performs NAdam algorithm computation.

        See :class:`~torch.optim.NAdam` for details.
        """

        kinetic_energy = 0
        for i, param in enumerate(params):
            grad = grads[i]
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            mu_product = mu_products[i]
            step = state_steps[i]

            bias_correction2 = 1 - beta2**step

            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # calculate the momentum cache \mu^{t} and \mu^{t+1}
            mu = beta1 * (1.0 - 0.5 * (0.96 ** (step * momentum_decay)))
            mu_next = beta1 * (1.0 - 0.5 * (0.96 ** ((step + 1) * momentum_decay)))
            mu_product = mu_product * mu
            mu_product_next = mu_product * mu * mu_next

            # decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            denom = exp_avg_sq.div(bias_correction2).sqrt().add_(eps)
            param.addcdiv_(grad, denom, value=-lr * (1.0 - mu) / (1.0 - mu_product))
            param.addcdiv_(exp_avg, denom, value=-lr * mu_next / (1.0 - mu_product_next))

            if output_info:
                kinetic_energy += lr * torch.sum(torch.sqrt((exp_avg**2) / ((1 - beta1) ** 2) + 1))

        if output_info:
            info_dict = {"kinetic_energy": kinetic_energy.item()}
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
            mu_products = []
            state_steps = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError("NAdam does not support sparse gradients")
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state["step"] = 0
                        state["mu_product"] = 1.0
                        # Exponential moving average of gradient values
                        state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])
                    mu_products.append(state["mu_product"])

                    # update the steps for each param group update
                    state["step"] += 1
                    # record the step after step update
                    state_steps.append(state["step"])

            info_dict = self.nadam(
                params=params_with_grad,
                grads=grads,
                exp_avgs=exp_avgs,
                exp_avg_sqs=exp_avg_sqs,
                mu_products=mu_products,
                state_steps=state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum_decay=group["momentum_decay"],
                eps=group["eps"],
                output_info=group["output_info"],
            )

            # update mu_product
            for p, mu_product in zip(params_with_grad, mu_products):
                state = self.state[p]
                state["mu_product"] = (
                    state["mu_product"] * beta1 * (1.0 - 0.5 * (0.96 ** (state["step"] * group["momentum_decay"])))
                )

        if group["output_info"]:
            return loss, info_dict
        else:
            return loss
