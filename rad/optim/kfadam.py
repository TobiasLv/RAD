import torch
from torch.optim.optimizer import Optimizer


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
