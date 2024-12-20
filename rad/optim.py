import math
from typing import List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer, required


class RAD(Optimizer):
    r"""Implements relativistic adaptive gradient descent algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        delta (float, optional): speed coefficient, strength of step size
            limitation (default: 0)
        order (int, optional): precision of the approximation to the relativistic
            Hamiltonian system (default: 1)
        max_iter (int, optional): the maximum iteration number for training,
          used to control the increasing sequence of eps_k to reach its highest
          value 1; if set as None, then zeta_k will anneal as 1-beta2^{k+1}
          (default: None, suggested: max_iter)
        weight_decay (float, optional): weight decay coefficient (default: 0)
        zeta (float, optional): the symplectic coefficient, None for annealing
          as 1-beta2^{k+1}, positive value for fixed zeta (default: None)
        bound_lr (float, optional): limit the upper and lower bounds of lr (default: None)
        final_delta (float, optional): the final value of delta, used to control
          the final value of delta, None for fixed delta (default: None)
        momentum_decay (float, optional): the decay rate of the momentum term,
         only used when nesterov=True (default: 4e-3)
        amsgrad (boolean, optional): whether to use the AMSGrad variant (default: False)
        nesterov (boolean, optional): whether to use the Nesterov momentum (default: False)
        output_info (boolean, optional): whether to output the information of
          the training process (default: False)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        delta=1,
        order=1,
        weight_decay=0,
        momentum_decay=4e-3,
        max_iter=None,
        zeta=None,
        bound_lr=None,
        final_delta=None,
        amsgrad=False,
        nesterov=False,
        output_info=False,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if delta < 0.0:
            raise ValueError("Invalid delta value: {}".format(delta))
        if order not in [1, 2]:
            raise ValueError("Invalid order order: {}".format(order))
        if max_iter is not None:
            if not 0 < max_iter:
                raise ValueError("Invalid max_iter value: {}".format(max_iter))
        if zeta is not None:
            if not 0.0 < zeta:
                raise ValueError("Invalid epsilon value: {}".format(zeta))
        if bound_lr is not None:
            if not 0.0 < bound_lr:
                raise ValueError("Invalid bound_lr value: {}".format(bound_lr))
        if final_delta is not None:
            if not 0.0 < final_delta:
                raise ValueError("Invalid final_delta value: {}".format(final_delta))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= momentum_decay:
            raise ValueError("Invalid momentum_decay value: {}".format(momentum_decay))

        defaults = dict(
            lr=lr,
            base_lr=lr,
            betas=betas,
            delta=delta,
            order=order,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
            max_iter=max_iter,
            zeta=zeta,
            bound_lr=bound_lr,
            final_delta=final_delta,
            amsgrad=amsgrad,
            nesterov=nesterov,
            output_info=output_info,
        )
        super(RAD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(RAD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("max_iter", None)
            group.setdefault("zeta", None)
            group.setdefault("bound_lr", None)
            group.setdefault("final_delta", None)
            group.setdefault("amsgrad", False)
            group.setdefault("nesterov", False)
            group.setdefault("output_info", False)

    def zeta_annealing(self, step, max_iter):
        if max_iter is None:
            eps = 1
        else:
            # recommand for RL
            exponent = 40 * (step / max_iter - 1)
            eps = math.exp(exponent) if exponent < 0 else 1

            # recommand for SL
            # exponent =  16 * (step / max_iter - 1) / (1/3)
            # eps = np.clip(10 ** exponent, 1e-16, 1)

        zeta = np.clip(eps, 1e-16, 1)
        return zeta

    def delta_annealing(self, step, max_iter, delta, final_delta):
        if max_iter is not None:
            # Warm up - Stay - Decay
            warm_up_ratio = 0.1
            decay_ratio = 0.1
            if step < max_iter * warm_up_ratio:
                delta = 1 / ((step) / (max_iter * warm_up_ratio) * (1 / delta - 1 / final_delta) + 1 / final_delta)
            elif step > max_iter * (1 - decay_ratio):
                delta = 1 / (
                    (1 - np.sqrt((step - max_iter * (1 - decay_ratio)) / (max_iter * decay_ratio)))
                    * (1 / delta - 1 / final_delta)
                    + 1 / final_delta
                )
            else:
                delta = delta
        return delta

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
                    raise RuntimeError("RAD does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["mu_product"] = 1.0
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                mu_product = state["mu_product"]
                beta1, beta2 = group["betas"]
                delta = group["delta"]
                lr = group["lr"]
                base_lr = group["base_lr"]
                order = group["order"]
                weight_decay = group["weight_decay"]
                momentum_decay = group["momentum_decay"]
                bound_lr = group["bound_lr"]
                final_delta = group["final_delta"]
                max_iter = group["max_iter"]
                zeta = group["zeta"]

                # Perform stepweight decay
                p.mul_(1 - lr * weight_decay)

                state["step"] += 1

                if group["nesterov"]:
                    # calculate the momentum cache \mu^{t} and \mu^{t+1}
                    mu = beta1 * (1.0 - 0.5 * (0.96 ** (state["step"] * momentum_decay)))
                    mu_next = beta1 * (1.0 - 0.5 * (0.96 ** ((state["step"] + 1) * momentum_decay)))
                    mu_product = mu_product * mu
                    mu_product_next = mu_product * mu * mu_next

                # zeta annealing
                if zeta is None:
                    zeta = self.zeta_annealing(state["step"], max_iter)

                # delta annealing
                if final_delta is not None:
                    delta = self.delta_annealing(state["step"], max_iter, delta, final_delta)

                # Decay the first and second moment running average coefficient
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # parameter update
                if group["amsgrad"]:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.maximum(state["max_exp_avg_sq"], exp_avg_sq, out=state["max_exp_avg_sq"])
                    # Use the max. for normalizing running avg. of gradient
                    denom = 1 / torch.sqrt(state["max_exp_avg_sq"] * (delta**2) * 4 + 4 * zeta)
                    if order == 1:
                        denom *= 2
                    elif order == 2:
                        denom += 1 / torch.sqrt(state["max_exp_avg_sq"] * (delta**2) * 4 + 4 * zeta / (beta1**2))
                else:
                    denom = 1 / torch.sqrt(exp_avg_sq * (delta**2) * 4 + 4 * zeta)
                    if order == 1:
                        denom *= 2
                    elif order == 2:
                        denom += 1 / torch.sqrt(exp_avg_sq * (delta**2) * 4 + 4 * zeta / (beta1**2))
                denom *= math.sqrt(bias_correction2)

                if bound_lr is not None:
                    final_lr = bound_lr * lr / base_lr
                    lower_bound = final_lr * (1 - 1 / ((1 - beta2) * state["step"] + 1))
                    upper_bound = final_lr * (1 + 1 / ((1 - beta2) * state["step"]))
                    if group["nesterov"]:
                        step_size = torch.full_like(denom, lr * (1.0 - mu) / (1.0 - mu_product))
                        step_size.mul_(denom).clamp_(lower_bound, upper_bound).mul_(grad)
                        p.add_(step_size, alpha=-1)
                        step_size = torch.full_like(denom, lr * mu_next / (1.0 - mu_product_next))
                    else:
                        step_size = torch.full_like(denom, lr / bias_correction1)
                    step_size.mul_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)
                    p.add_(step_size, alpha=-1)
                else:
                    if group["nesterov"]:
                        p.addcmul_(grad, denom, value=-lr * (1.0 - mu) / (1.0 - mu_product))
                        p.addcmul_(exp_avg, denom, value=-lr * mu_next / (1.0 - mu_product_next))
                    else:
                        p.addcmul_(exp_avg, denom, value=-lr / bias_correction1)

                if group["nesterov"]:
                    # update mu_product
                    state["mu_product"] = (
                        state["mu_product"] * beta1 * (1.0 - 0.5 * (0.96 ** (state["step"] * momentum_decay)))
                    )

                if group["output_info"]:
                    kinetic_energy += (
                        lr / delta * torch.sum(torch.sqrt((exp_avg**2) / ((1 - beta1) ** 2) + 1 / (delta**2)))
                    )
                    exp_avg_norm += torch.sum(exp_avg**2)
                    exp_avg_sq_norm += torch.sum(exp_avg_sq**2)
                    efficient_lr = denom * lr / bias_correction1
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


class SGD(Optimizer):
    r"""Implements stochastic gradient descent.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0),
            when momentum is larger than 0, Heavy-ball (HB) methods are implemented
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        symplectic (bool, optional): whether to use symplectic update (default: True)
        output_info (boolean, optional): whether to output the information of
          the training process (default: False)
    """

    def __init__(self, params, lr=required, momentum=0, weight_decay=0, symplectic=True, output_info=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            symplectic=symplectic,
            output_info=output_info,
        )
        super(SGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("symplectic", True)
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
            symplectic = group["symplectic"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.zeros_like(d_p).detach()
                        if not symplectic:
                            d_p_tmp = buf
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        if not symplectic:
                            d_p_tmp = buf
                        buf.mul_(momentum).add_(d_p, alpha=1 - momentum)
                    if not symplectic:
                        d_p = d_p_tmp
                    else:
                        d_p = buf
                p.add_(d_p, alpha=-lr)

                exp_avg_norm_sq_total += torch.norm(d_p) ** 2

        kinetic_energy = exp_avg_norm_sq_total * lr / (2 * (1 - momentum))

        if group["output_info"]:
            info_dict = {"kinetic_energy": kinetic_energy.item()}
            return loss, info_dict
        else:
            return loss


class NAG(Optimizer):
    r"""Implements Nesterov's accelerated gradient.

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
        if momentum <= 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, output_info=output_info)
        super(NAG, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(NAG, self).__setstate__(state)
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
                    d_p = buf.mul(momentum).add(d_p, alpha=1 - momentum) / 2
                p.add_(d_p, alpha=-lr)

                exp_avg_norm_sq_total += torch.norm(buf) ** 2

        kinetic_energy = exp_avg_norm_sq_total * lr / (2 * (1 - momentum))

        if group["output_info"]:
            info_dict = {"kinetic_energy": kinetic_energy.item()}
            return loss, info_dict
        else:
            return loss


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
        if momentum <= 0.0:
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
                p.add_(d_p, alpha=-lr)

                exp_avg_norm_sq_total += torch.norm(buf) ** 2

        kinetic_energy = exp_avg_norm_sq_total * lr / (2 * (1 - momentum))

        if group["output_info"]:
            info_dict = {"kinetic_energy": kinetic_energy.item()}
            return loss, info_dict
        else:
            return loss


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


class SWATS(Optimizer):
    r"""Implements Switching from Adam to SGD technique.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant (default: False)
        verbose (boolean, optional): whether to print switching information (default: False)
        nesterov (boolean, optional): whether to use the Nesterov momentum (default: False)
        output_info (boolean, optional): whether to output the information of
          the training process (default: False)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        verbose=False,
        nesterov=False,
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
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            phase="ADAM",
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            verbose=verbose,
            nesterov=nesterov,
            output_info=output_info,
        )

        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("nesterov", False)
            group.setdefault("verbose", False)
            group.setdefault("output_info", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional):
                A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        kinetic_energy = 0
        for group in self.param_groups:
            for w in group["params"]:
                if w.grad is None:
                    continue
                grad = w.grad.data

                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, " "please consider SparseAdam instead")

                amsgrad = group["amsgrad"]

                state = self.state[w]

                # state initialization
                if len(state) == 0:
                    state["step"] = 0
                    # exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(w.data)
                    # exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(w.data)
                    # moving average for the non-orthogonal projection scaling
                    state["exp_avg2"] = w.new(1).fill_(0)
                    if amsgrad:
                        # maintains max of all exp. moving avg.
                        # of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(w.data)

                exp_avg, exp_avg2, exp_avg_sq = (
                    state["exp_avg"],
                    state["exp_avg2"],
                    state["exp_avg_sq"],
                )

                if amsgrad:
                    max_exp_avg_sq = state["max_exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                if group["weight_decay"] != 0:
                    grad.add_(group["weight_decay"], w.data)

                # if its SGD phase, take an SGD update and continue
                if group["phase"] == "SGD":
                    if "momentum_buffer" not in state:
                        buf = state["momentum_buffer"] = torch.clone(grad).detach()
                    else:
                        buf = state["momentum_buffer"]
                        buf.mul_(beta1).add_(grad)
                        grad = buf

                    grad.mul_(1 - beta1)
                    if group["nesterov"]:
                        grad.add_(beta1, buf)

                    w.data.add_(-group["lr"], grad)
                    kinetic_energy += torch.norm(grad) ** 2 * group["lr"] / (2 * (1 - beta1))
                    continue

                # decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # maintains the maximum of all 2nd
                    # moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group["eps"])
                else:
                    denom = exp_avg_sq.sqrt().add_(group["eps"])

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]
                step_size = group["lr"] * (bias_correction2**0.5) / bias_correction1

                p = -step_size * (exp_avg / denom)
                w.data.add_(p)

                p_view = p.view(-1)
                pg = p_view.dot(grad.view(-1))

                if pg != 0:
                    # the non-orthognal scaling estimate
                    scaling = p_view.dot(p_view) / -pg
                    exp_avg2.mul_(beta2).add_(1 - beta2, scaling)

                    # bias corrected exponential average
                    corrected_exp_avg = exp_avg2 / bias_correction2

                    # checking criteria of switching to SGD training
                    if state["step"] > 1 and corrected_exp_avg.allclose(scaling, rtol=1e-6) and corrected_exp_avg > 0:
                        group["phase"] = "SGD"
                        group["lr"] = corrected_exp_avg.item()
                        if group["verbose"]:
                            print(
                                "Switching to SGD after "
                                "{} steps with lr {:.5f} "
                                "and momentum {:.5f}.".format(state["step"], group["lr"], beta1)
                            )

                kinetic_energy += group["lr"] * torch.sum(torch.sqrt((exp_avg**2) / ((1 - beta1) ** 2) + 1))

        if group["output_info"]:
            info_dict = {"kinetic_energy": kinetic_energy.item()}
            return loss, info_dict
        else:
            return loss


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


class KFAdam(Optimizer):
    r"""Implements the KFAdam optimization algorithm. The gradient and the standard deviation
    are estimated using a Kalman Filter instead of an EMA filter.

    Arguments:
        params: iterable of parameters to optimize or dicts defining
            parameter groups
        lr: learning rate (default: 1e-3)
        beta: coefficient used for computing
            running averages of error variances (default: 0.95)
        eps: term added to the denominator to improve
            numerical stability (default: 1e-12)
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.95, 0, 999),
        eps: float = 1e-12,
        weight_decay: float = 0,
        output_info: bool = False,
    ):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not (0.0 < betas[0] < 1.0):
            raise ValueError("Invalid beta value: {}".format(betas[0]))
        if eps < 0.0:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = {
            "lr": lr,
            "beta": betas[0],
            "eps": eps,
            "weight_decay": weight_decay,
            "output_info": output_info,
        }
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure: A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        exp_avg_norm = 0
        exp_avg_sq_norm = 0
        efficient_lr_norm = 0
        step_size_norm = 0
        grad_norm = 0
        grad_sq_norm = 0
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["beta"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                where = grad != 0
                if grad.is_sparse:
                    raise RuntimeError(
                        "KFAdam does not support sparse gradients, " "please consider SparseAdam instead"
                    )
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    sigma_sq_init = grad[where].pow(2).mean()
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["obs_prev"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["obs_one_before_prev"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["estimate_error"] = sigma_sq_init
                    state["obs_prev_var"] = sigma_sq_init
                    state["obs_one_before_prev_var"] = sigma_sq_init

                state["step"] += 1

                state["obs_prev_var"] = (
                    state["obs_prev_var"] * beta + (1 - beta) * (grad - state["obs_prev"])[where].pow(2).mean()
                )
                state["obs_one_before_prev_var"] = (
                    state["obs_one_before_prev_var"] * beta
                    + (1 - beta) * (grad - state["obs_one_before_prev"])[where].pow(2).mean()
                )
                # Reference: http://article.nadiapub.com/IJCA/vol10_no10/6.pdf
                measurement_variance = (
                    (state["obs_prev_var"] - 0.5 * state["obs_one_before_prev_var"]) / (1.0 - beta ** state["step"])
                ).clamp(min=eps)
                process_variance = (
                    (state["obs_one_before_prev_var"] - state["obs_prev_var"]) / (1.0 - beta ** state["step"])
                ).clamp(min=eps)
                prediction = state["exp_avg"]
                prediction_error = state["estimate_error"] + process_variance
                kalman_gain = (prediction_error / (eps + prediction_error + measurement_variance)).clamp(
                    min=0.0, max=1.0
                )
                innovation = grad - prediction
                estimate = prediction + kalman_gain * innovation
                estimate_error = (1.0 - kalman_gain) * prediction_error
                state["estimate_error"] = estimate_error
                state["exp_avg"] = torch.where(where, estimate, state["exp_avg"])

                step = -lr * estimate / (torch.sqrt(estimate_error + estimate.pow(2)) + eps)
                step = torch.where(where, step, torch.zeros_like(step))

                # Perform stepweight decay
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(step)
                state["obs_one_before_prev"] = torch.where(where, state["obs_prev"], state["obs_one_before_prev"])
                state["obs_prev"] = torch.where(where, grad, state["obs_prev"])

                if group["output_info"]:
                    exp_avg_norm += torch.sum(estimate**2)
                    exp_avg_sq_norm += torch.sum(torch.sqrt(estimate_error + estimate.pow(2)) ** 2)
                    efficient_lr_norm += torch.sum((-lr / (torch.sqrt(estimate_error + estimate.pow(2)) + eps)) ** 2)
                    step_size_norm += torch.sum(step**2)
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
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

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

                p.data.addcmul_(-1 / bias_correction1, exp_avg, s2post)

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
