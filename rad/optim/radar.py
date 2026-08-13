import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class RADAR(Optimizer):
    r"""Implements the RADAR optimization algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        gamma (float, optional): gradient residual correction coefficient
            (default: 0.01)
        l (float, optional): residual correction step size. If None, it is set
            to 0.01 times the initial learning rate and remains fixed during
            training (default: None)
        delta (float, optional): scaling coefficient in the adaptive
            preconditioner (default: 1)
        zeta (float, optional): numerical stability coefficient
            (default: 1e-16)
        weight_decay (float, optional): weight decay coefficient
            (default: 0)
        decoupled_weight_decay (bool, optional): whether to use decoupled
            weight decay (default: True)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        gamma=0.01,
        l=None,
        delta=1,
        zeta=1e-16,
        weight_decay=0,
        decoupled_weight_decay=True,
    ):
        if lr < 0.0:
            raise ValueError(
                "Invalid learning rate: {}".format(lr)
            )

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 0: {}".format(
                    betas[0]
                )
            )

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter at index 1: {}".format(
                    betas[1]
                )
            )

        if gamma < 0.0:
            raise ValueError(
                "Invalid gamma value: {}".format(gamma)
            )

        if l is not None and l < 0.0:
            raise ValueError(
                "Invalid l value: {}".format(l)
            )

        if delta <= 0.0:
            raise ValueError(
                "Invalid delta value: {}".format(delta)
            )

        if zeta <= 0.0:
            raise ValueError(
                "Invalid zeta value: {}".format(zeta)
            )

        if weight_decay < 0.0:
            raise ValueError(
                "Invalid weight_decay value: {}".format(
                    weight_decay
                )
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            gamma=gamma,
            l=l,
            delta=delta,
            zeta=zeta,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
        )

        super().__init__(params, defaults)

    def add_param_group(self, param_group):
        """Add a parameter group and initialize its fixed residual step size."""
        super().add_param_group(param_group)

        group = self.param_groups[-1]
        if group["l"] is None:
            # Default l = 0.01 * the group's initial learning rate.
            # It remains fixed when lr is changed by a scheduler.
            group["l"] = 0.01 * group["lr"]

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("gamma", 0.01)
            group.setdefault("delta", 1)
            group.setdefault("zeta", 1e-16)
            group.setdefault("weight_decay", 0)
            group.setdefault(
                "decoupled_weight_decay",
                True,
            )

            if "l" not in group or group["l"] is None:
                group["l"] = 0.01 * group["lr"]

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): a closure that reevaluates the model
                and returns the loss.
        """

        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            gamma = group["gamma"]
            l = group["l"]
            delta = group["delta"]
            zeta = group["zeta"]
            weight_decay = group["weight_decay"]
            decoupled_weight_decay = group[
                "decoupled_weight_decay"
            ]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad: Tensor = p.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "RADAR does not support sparse gradients"
                    )

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

                    state["prev_grad"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]
                prev_grad: Tensor = state["prev_grad"]

                state["step"] += 1

                # Coupled weight decay
                if (
                    weight_decay != 0
                    and not decoupled_weight_decay
                ):
                    grad = grad.add(
                        p,
                        alpha=weight_decay,
                    )

                # Bias correction
                bias_correction1 = (
                    1 - beta1 ** state["step"]
                )

                bias_correction2 = (
                    1 - beta2 ** state["step"]
                )

                # First-order momentum
                #
                # m_t =
                # beta1 * m_{t-1}
                # + (1 - beta1) * g_t
                # + gamma * (g_t - g_{t-1})
                exp_avg.mul_(beta1).add_(
                    grad,
                    alpha=1 - beta1 + gamma,
                )

                exp_avg.add_(
                    prev_grad,
                    alpha=-gamma,
                )

                # Second moment
                #
                # v_t =
                # beta2 * v_{t-1}
                # + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # Adaptive preconditioner
                #
                # v_hat =
                # v_t / (1 - beta2^t)
                #
                # denom =
                # 1 / sqrt(delta^2 * v_hat + zeta)
                denom: Tensor = 1 / torch.sqrt(
                    delta ** 2
                    * exp_avg_sq
                    / bias_correction2
                    + zeta
                )

                # RADAR parameter update
                #
                # m_hat =
                # m_t / (1 - beta1^t)
                #
                # theta_{t+1} =
                # theta_t
                # - lr * denom * m_hat
                # + l  * denom * m_hat
                # - l  * denom * g_t

                p.addcmul_(
                    exp_avg,
                    denom,
                    value=-lr / bias_correction1,
                )

                p.addcmul_(
                    exp_avg,
                    denom,
                    value=l / bias_correction1,
                )

                p.addcmul_(
                    grad,
                    denom,
                    value=-l,
                )

                # Decoupled weight decay
                if (
                    weight_decay != 0
                    and decoupled_weight_decay
                ):
                    p.mul_(
                        1 - lr * weight_decay
                    )

                # Store current gradient
                prev_grad.copy_(grad)

        return loss
