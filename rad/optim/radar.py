import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer


class RADAR(Optimizer):
    r"""Implements the RADAR optimization algorithm.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.

        lr (float, optional):
            Learning rate. Default: 1e-3.

        betas (Tuple[float, float], optional):
            Coefficients used for computing running averages of
            gradient and its square. Default: (0.9, 0.999).

        gamma (float, optional):
            Gradient residual correction coefficient. Default: 0.01.

        l (float, optional):
            Residual correction step size. If l is None, l is initialized as
            0.01 * initial_lr for each parameter group and remains fixed when
            a learning-rate scheduler changes lr.

        delta (float, optional):
            Scaling coefficient in the adaptive preconditioner. Default: 1.

        zeta (float, optional):
            Numerical stability coefficient. Default: 1e-16.

        weight_decay (float, optional):
            Weight decay coefficient. Default: 0.

        decoupled_weight_decay (bool, optional):
            Whether to use AdamW-style decoupled weight decay. Default: True.

        foreach (bool, optional):
            Whether to use the multi-tensor foreach implementation.
            Default: True.
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        gamma=0.01,
        l=None,
        delta=1.0,
        zeta=1e-16,
        weight_decay=0.0,
        decoupled_weight_decay=True,
        *,
        foreach=True,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 0: {betas[0]}"
            )

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 1: {betas[1]}"
            )

        if gamma < 0.0:
            raise ValueError(f"Invalid gamma value: {gamma}")

        if l is not None and l < 0.0:
            raise ValueError(f"Invalid l value: {l}")

        if delta <= 0.0:
            raise ValueError(f"Invalid delta value: {delta}")

        if zeta <= 0.0:
            raise ValueError(f"Invalid zeta value: {zeta}")

        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        if not isinstance(foreach, bool):
            raise TypeError(f"foreach must be bool, got {type(foreach)}")

        defaults = dict(
            lr=lr,
            betas=betas,
            gamma=gamma,
            l=l,
            delta=delta,
            zeta=zeta,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            foreach=foreach,
        )

        super().__init__(params, defaults)

    def add_param_group(self, param_group):
        """Add a parameter group and initialize its fixed residual step size."""
        super().add_param_group(param_group)

        group = self.param_groups[-1]
        if group["l"] is None:
            group["l"] = 0.01 * group["lr"]

    # ================================================================
    # Utility / checkpoint migration
    # ================================================================

    @staticmethod
    def _step_to_int(step):
        """Convert an old checkpoint step representation to Python int."""
        if torch.is_tensor(step):
            if step.numel() != 1:
                raise RuntimeError(
                    "RADAR state['step'] Tensor must contain one element."
                )
            return int(step.detach().cpu().item())

        return int(step)

    @staticmethod
    def _migrate_legacy_param_state_(p_state, beta1, gamma):
        if "prev_grad" not in p_state:
            return

        prev_grad = p_state["prev_grad"]
        exp_avg = p_state.get("exp_avg", None)

        if exp_avg is None:
            p_state.pop("prev_grad", None)
            p_state.pop("prev_grad_valid", None)
            return

        if beta1 > 0.0:
            if gamma != 0.0:
                ratio = gamma / beta1
                ema_scale = 1.0 - ratio

                # If ema_scale == 0, the reconstructed corrected moment no
                # longer depends on the EMA, so the historical EMA value is
                # irrelevant for future updates. Reset it safely.
                if ema_scale == 0.0:
                    exp_avg.zero_()
                else:
                    exp_avg.add_(prev_grad, alpha=-ratio)
                    exp_avg.div_(ema_scale)

            # gamma == 0: legacy exp_avg is already the ordinary EMA.
            p_state.pop("prev_grad", None)
            p_state.pop("prev_grad_valid", None)

        else:
            # beta1 == 0 fallback stores previous effective gradient in
            # exp_avg, so copy legacy prev_grad into exp_avg.
            exp_avg.copy_(prev_grad)
            p_state.pop("prev_grad", None)
            p_state["prev_grad_valid"] = bool(
                p_state.get("prev_grad_valid", True)
            )

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:
            group.setdefault("gamma", 0.01)
            group.setdefault("l", None)
            if group["l"] is None:
                group["l"] = 0.01 * group["lr"]

            group.setdefault("delta", 1.0)
            group.setdefault("zeta", 1e-16)
            group.setdefault("weight_decay", 0.0)
            group.setdefault("decoupled_weight_decay", True)

            if group.get("foreach") is None:
                group["foreach"] = True

            beta1, _ = group["betas"]
            gamma = group["gamma"]

            for p in group["params"]:
                p_state = self.state.get(p, None)
                if not p_state:
                    continue

                if "step" in p_state:
                    p_state["step"] = self._step_to_int(p_state["step"])

                self._migrate_legacy_param_state_(
                    p_state,
                    beta1,
                    gamma,
                )

                if beta1 == 0.0:
                    p_state.setdefault("prev_grad_valid", True)
                else:
                    p_state.pop("prev_grad_valid", None)

    @staticmethod
    def _single_tensor_radar_reparameterized(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        steps,
        *,
        lr,
        beta1,
        beta2,
        gamma,
        l,
        delta,
        zeta,
        weight_decay,
        decoupled_weight_decay,
    ):
        delta_sq = delta * delta
        gamma_ratio = gamma / beta1
        ema_coefficient = 1.0 - gamma_ratio

        for p, grad, exp_avg, exp_avg_sq, step in zip(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            steps,
        ):
            # ----------------------------------------------------------
            # Weight decay
            # ----------------------------------------------------------
            if weight_decay != 0.0:
                if decoupled_weight_decay:
                    p.mul_(1.0 - lr * weight_decay)
                else:
                    grad = grad.add(p, alpha=weight_decay)

            # ----------------------------------------------------------
            # Bias correction
            # ----------------------------------------------------------
            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step

            # ----------------------------------------------------------
            # Ordinary EMA only:
            # mbar_t = beta1*mbar_{t-1} + (1-beta1)*g_t
            # ----------------------------------------------------------
            exp_avg.lerp_(grad, 1.0 - beta1)

            # ----------------------------------------------------------
            # Second moment
            # ----------------------------------------------------------
            exp_avg_sq.mul_(beta2)
            exp_avg_sq.addcmul_(grad, grad, value=1.0 - beta2)

            # ----------------------------------------------------------
            # denominator = sqrt(v_t + zeta * bc2 / delta^2)
            # ----------------------------------------------------------
            denominator = exp_avg_sq.add(
                zeta * bias_correction2 / delta_sq
            )
            denominator.sqrt_()

            preconditioner_scale = bias_correction2 ** 0.5 / delta

            momentum_step_size = (
                -(lr - l)
                / bias_correction1
                * preconditioner_scale
                * ema_coefficient
            )

            gradient_step_size = (
                -preconditioner_scale
                * (
                    l
                    + (lr - l)
                    / bias_correction1
                    * gamma_ratio
                )
            )

            p.addcdiv_(
                exp_avg,
                denominator,
                value=momentum_step_size,
            )

            p.addcdiv_(
                grad,
                denominator,
                value=gradient_step_size,
            )

    # ================================================================
    # beta1 == 0 fallback
    # ================================================================

    @staticmethod
    def _single_tensor_radar_beta1_zero(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        prev_grad_valids,
        steps,
        *,
        lr,
        beta2,
        gamma,
        l,
        delta,
        zeta,
        weight_decay,
        decoupled_weight_decay,
    ):
        """Exact beta1 == 0 fallback using exp_avg as previous gradient."""
        delta_sq = delta * delta

        for (
            p,
            grad,
            prev_grad,
            exp_avg_sq,
            prev_grad_valid,
            step,
        ) in zip(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            prev_grad_valids,
            steps,
        ):
            if weight_decay != 0.0:
                if decoupled_weight_decay:
                    p.mul_(1.0 - lr * weight_decay)
                else:
                    grad = grad.add(p, alpha=weight_decay)

            bias_correction2 = 1.0 - beta2 ** step

            exp_avg_sq.mul_(beta2)
            exp_avg_sq.addcmul_(grad, grad, value=1.0 - beta2)

            denominator = exp_avg_sq.add(
                zeta * bias_correction2 / delta_sq
            )
            denominator.sqrt_()

            preconditioner_scale = bias_correction2 ** 0.5 / delta

            if prev_grad_valid and gamma != 0.0:
                # beta1 = 0:
                # corrected m_t = (1 + gamma) * g_t - gamma * g_{t-1}
                prev_grad_step_size = (
                    (lr - l)
                    * gamma
                    * preconditioner_scale
                )

                gradient_step_size = (
                    -preconditioner_scale
                    * (
                        (lr - l) * (1.0 + gamma)
                        + l
                    )
                )

                p.addcdiv_(
                    prev_grad,
                    denominator,
                    value=prev_grad_step_size,
                )

                p.addcdiv_(
                    grad,
                    denominator,
                    value=gradient_step_size,
                )
            else:
                # Initial/recovered no-residual step when legacy semantics
                # require the residual correction to be skipped.
                p.addcdiv_(
                    grad,
                    denominator,
                    value=-lr * preconditioner_scale,
                )

            prev_grad.copy_(grad)

    # ================================================================
    # beta1 > 0: foreach implementation for one device/dtype bucket
    # ================================================================

    @staticmethod
    def _foreach_bucket_reparameterized(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        indices,
        steps,
        uniform_step,
        uniform_step_value,
        *,
        lr,
        beta1,
        beta2,
        gamma,
        l,
        delta,
        zeta,
        weight_decay,
        decoupled_weight_decay,
    ):
        if len(params) == 0:
            return

        # ------------------------------------------------------------
        # Weight decay
        # ------------------------------------------------------------
        if weight_decay != 0.0:
            if decoupled_weight_decay:
                torch._foreach_mul_(
                    params,
                    1.0 - lr * weight_decay,
                )
            else:
                grads = list(
                    torch._foreach_add(
                        grads,
                        params,
                        alpha=weight_decay,
                    )
                )

        # ------------------------------------------------------------
        # Ordinary EMA only.  No prev_grad and no residual kernels.
        # ------------------------------------------------------------
        torch._foreach_lerp_(
            exp_avgs,
            grads,
            1.0 - beta1,
        )

        # ------------------------------------------------------------
        # Second moment
        # ------------------------------------------------------------
        torch._foreach_mul_(
            exp_avg_sqs,
            beta2,
        )

        torch._foreach_addcmul_(
            exp_avg_sqs,
            grads,
            grads,
            1.0 - beta2,
        )

        delta_sq = delta * delta
        gamma_ratio = gamma / beta1
        ema_coefficient = 1.0 - gamma_ratio

        # ------------------------------------------------------------
        # Fast path: same active-step counter for all tensors in bucket.
        # ------------------------------------------------------------
        if uniform_step:
            bias_correction1 = 1.0 - beta1 ** uniform_step_value
            bias_correction2 = 1.0 - beta2 ** uniform_step_value

            denominator_shift = (
                zeta * bias_correction2 / delta_sq
            )

            denominators = list(
                torch._foreach_add(
                    exp_avg_sqs,
                    denominator_shift,
                )
            )
            torch._foreach_sqrt_(denominators)

            preconditioner_scale = (
                bias_correction2 ** 0.5 / delta
            )

            momentum_step_size = (
                -(lr - l)
                / bias_correction1
                * preconditioner_scale
                * ema_coefficient
            )

            gradient_step_size = (
                -preconditioner_scale
                * (
                    l
                    + (lr - l)
                    / bias_correction1
                    * gamma_ratio
                )
            )

            torch._foreach_addcdiv_(
                params,
                exp_avgs,
                denominators,
                momentum_step_size,
            )

            torch._foreach_addcdiv_(
                params,
                grads,
                denominators,
                gradient_step_size,
            )

        # ------------------------------------------------------------
        # Rare path: different active-step counters because some params
        # have had grad=None at earlier optimizer steps.
        # ------------------------------------------------------------
        else:
            device_steps = [steps[index] for index in indices]

            bias_correction1 = [
                1.0 - beta1 ** step
                for step in device_steps
            ]

            bias_correction2 = [
                1.0 - beta2 ** step
                for step in device_steps
            ]

            denominator_shifts = [
                zeta * bc2 / delta_sq
                for bc2 in bias_correction2
            ]

            denominators = list(
                torch._foreach_add(
                    exp_avg_sqs,
                    denominator_shifts,
                )
            )
            torch._foreach_sqrt_(denominators)

            preconditioner_scales = [
                bc2 ** 0.5 / delta
                for bc2 in bias_correction2
            ]

            momentum_step_sizes = [
                -(
                    (lr - l)
                    / bc1
                    * scale
                    * ema_coefficient
                )
                for bc1, scale in zip(
                    bias_correction1,
                    preconditioner_scales,
                )
            ]

            gradient_step_sizes = [
                -scale
                * (
                    l
                    + (lr - l)
                    / bc1
                    * gamma_ratio
                )
                for bc1, scale in zip(
                    bias_correction1,
                    preconditioner_scales,
                )
            ]

            torch._foreach_addcdiv_(
                params,
                exp_avgs,
                denominators,
                momentum_step_sizes,
            )

            torch._foreach_addcdiv_(
                params,
                grads,
                denominators,
                gradient_step_sizes,
            )

        del denominators

    # ================================================================
    # Multi-tensor foreach dispatch
    # ================================================================

    @classmethod
    def _multi_tensor_radar_reparameterized(
        cls,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        steps,
        uniform_step,
        uniform_step_value,
        **kwargs,
    ):
        grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
            [
                params,
                grads,
                exp_avgs,
                exp_avg_sqs,
            ],
            with_indices=True,
        )

        for device_tensor_lists, indices in grouped_tensors.values():
            (
                device_params,
                device_grads,
                device_exp_avgs,
                device_exp_avg_sqs,
            ) = device_tensor_lists

            cls._foreach_bucket_reparameterized(
                device_params,
                device_grads,
                device_exp_avgs,
                device_exp_avg_sqs,
                indices,
                steps,
                uniform_step,
                uniform_step_value,
                **kwargs,
            )

    # ================================================================
    # Main optimizer step
    # ================================================================

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]

            params = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            steps = []

            # Only used by the beta1 == 0 fallback.
            prev_grad_valids = []

            uniform_step = True
            first_step = None

            # --------------------------------------------------------
            # Gather active parameters
            # --------------------------------------------------------
            for p in group["params"]:
                grad: Tensor = p.grad

                if grad is None:
                    # beta1 > 0 optimized path treats missing optimizer steps
                    # as absent observations in the active-gradient sequence.
                    # beta1 == 0 fallback can preserve the old one-step skip.
                    if beta1 == 0.0:
                        state = self.state.get(p, None)
                        if state:
                            state["prev_grad_valid"] = False
                    continue

                if grad.is_sparse:
                    raise RuntimeError(
                        "RADAR does not support sparse gradients."
                    )

                if torch.is_complex(p):
                    raise RuntimeError(
                        "RADAR does not currently support complex parameters."
                    )

                state = self.state[p]

                # ----------------------------------------------------
                # Lazy initialization
                # ----------------------------------------------------
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

                    if beta1 == 0.0:
                        # In this rare fallback exp_avg stores prev_grad.
                        state["prev_grad_valid"] = True

                else:
                    # Safety for states loaded without __setstate__ migration.
                    if "prev_grad" in state:
                        self._migrate_legacy_param_state_(
                            state,
                            beta1,
                            group["gamma"],
                        )

                    state["step"] = self._step_to_int(state["step"])

                if beta1 == 0.0:
                    prev_grad_valids.append(
                        bool(state.get("prev_grad_valid", True))
                    )

                state["step"] += 1
                step = state["step"]

                if first_step is None:
                    first_step = step
                elif step != first_step:
                    uniform_step = False

                params.append(p)
                grads.append(grad)
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps.append(step)

                if beta1 == 0.0:
                    state["prev_grad_valid"] = True

            if len(params) == 0:
                continue

            kwargs = dict(
                lr=group["lr"],
                beta2=beta2,
                gamma=group["gamma"],
                l=group["l"],
                delta=group["delta"],
                zeta=group["zeta"],
                weight_decay=group["weight_decay"],
                decoupled_weight_decay=group[
                    "decoupled_weight_decay"
                ],
            )

            # --------------------------------------------------------
            # beta1 == 0: exact rare fallback.
            # --------------------------------------------------------
            if beta1 == 0.0:
                self._single_tensor_radar_beta1_zero(
                    params,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    prev_grad_valids,
                    steps,
                    **kwargs,
                )
                continue

            # --------------------------------------------------------
            # beta1 > 0: optimized reparameterized RADAR.
            # --------------------------------------------------------
            kwargs["beta1"] = beta1

            if group["foreach"]:
                self._multi_tensor_radar_reparameterized(
                    params,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    steps,
                    uniform_step,
                    first_step,
                    **kwargs,
                )
            else:
                self._single_tensor_radar_reparameterized(
                    params,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    steps,
                    **kwargs,
                )

        return loss
