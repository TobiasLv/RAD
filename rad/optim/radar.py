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
            Falls back to the single-tensor path when unavailable.
            Default: True.

    Note:
        Parameters stored directly as torch.float16 are not supported. Use
        automatic mixed precision with FP32 parameters. BF16 may be used when
        it is supported by the installed PyTorch version and device.
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
            raise RuntimeError(
                "Invalid RADAR checkpoint: state contains prev_grad "
                "but no exp_avg."
            )

        if beta1 == 0.0:
            exp_avg.copy_(prev_grad)
        elif gamma != 0.0:
            ema_scale = beta1 - gamma

            # When beta1 == gamma, the corrected moment no longer depends on
            # the ordinary EMA, so that historical EMA cannot be recovered
            # and is irrelevant while the hyperparameters remain fixed.
            if ema_scale == 0.0:
                exp_avg.zero_()
            else:
                exp_avg.mul_(beta1)
                exp_avg.add_(prev_grad, alpha=-gamma)
                exp_avg.div_(ema_scale)

        # gamma == 0: legacy exp_avg is already the ordinary EMA.
        p_state.pop("prev_grad", None)
        p_state.pop("prev_grad_valid", None)

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
                group["foreach"] = self.defaults.get("foreach", True)

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

                p_state.pop("prev_grad_valid", None)

    @staticmethod
    def _can_use_foreach():
        return callable(
            getattr(
                Optimizer,
                "_group_tensors_by_device_and_dtype",
                None,
            )
        )

    @staticmethod
    def _single_tensor_radar_reparameterized(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
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
        delta_sq = delta * delta
        ema_alpha = 1.0 - beta1
        second_moment_alpha = 1.0 - beta2
        previous_ema_coefficient = beta1 - gamma
        gradient_coefficient = 1.0 - beta1 + gamma
        decay_factor = 1.0 - lr * weight_decay

        if uniform_step:
            bias_correction1 = 1.0 - beta1 ** uniform_step_value
            bias_correction2 = 1.0 - beta2 ** uniform_step_value
            denominator_shift = zeta * bias_correction2 / delta_sq
            preconditioner_scale = bias_correction2 ** 0.5 / delta
            base_step_size = (
                -(lr - l)
                / bias_correction1
                * preconditioner_scale
            )
            previous_ema_step_size = (
                base_step_size * previous_ema_coefficient
            )
            gradient_step_size = (
                base_step_size * gradient_coefficient
                - l * preconditioner_scale
            )

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
                    p.mul_(decay_factor)
                else:
                    grad = grad.add(p, alpha=weight_decay)

            if not uniform_step:
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denominator_shift = zeta * bias_correction2 / delta_sq
                preconditioner_scale = bias_correction2 ** 0.5 / delta
                base_step_size = (
                    -(lr - l)
                    / bias_correction1
                    * preconditioner_scale
                )
                previous_ema_step_size = (
                    base_step_size * previous_ema_coefficient
                )
                gradient_step_size = (
                    base_step_size * gradient_coefficient
                    - l * preconditioner_scale
                )

            # ----------------------------------------------------------
            # Second moment
            # ----------------------------------------------------------
            exp_avg_sq.mul_(beta2)
            exp_avg_sq.addcmul_(grad, grad, value=second_moment_alpha)

            # ----------------------------------------------------------
            # denominator = sqrt(v_t + zeta * bc2 / delta^2)
            # ----------------------------------------------------------
            denominator = exp_avg_sq.add(denominator_shift)
            denominator.sqrt_()

            p.addcdiv_(
                exp_avg,
                denominator,
                value=previous_ema_step_size,
            )

            p.addcdiv_(
                grad,
                denominator,
                value=gradient_step_size,
            )

            # Store only the ordinary EMA. The corrected RADAR moment used
            # above is reconstructed from the previous EMA and current grad.
            exp_avg.lerp_(grad, ema_alpha)

    # ================================================================
    # Foreach implementation for one device/dtype bucket
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

            base_step_size = (
                -(lr - l)
                / bias_correction1
                * preconditioner_scale
            )

            previous_ema_step_size = (
                base_step_size * (beta1 - gamma)
            )

            gradient_step_size = (
                base_step_size * (1.0 - beta1 + gamma)
                - l * preconditioner_scale
            )

            torch._foreach_addcdiv_(
                params,
                exp_avgs,
                denominators,
                previous_ema_step_size,
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

            base_step_sizes = [
                -(
                    (lr - l)
                    / bc1
                    * scale
                )
                for bc1, scale in zip(
                    bias_correction1,
                    preconditioner_scales,
                )
            ]

            previous_ema_step_sizes = [
                base * (beta1 - gamma)
                for base in base_step_sizes
            ]

            gradient_step_sizes = [
                base * (1.0 - beta1 + gamma)
                - l * scale
                for base, scale in zip(
                    base_step_sizes,
                    preconditioner_scales,
                )
            ]

            torch._foreach_addcdiv_(
                params,
                exp_avgs,
                denominators,
                previous_ema_step_sizes,
            )

            torch._foreach_addcdiv_(
                params,
                grads,
                denominators,
                gradient_step_sizes,
            )

        torch._foreach_lerp_(
            exp_avgs,
            grads,
            1.0 - beta1,
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
            with_indices=not uniform_step,
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

        # Gather and validate every active parameter before updating state or
        # parameters. The gathered lists are reused by the update phase so
        # each parameter group is scanned only once.
        prepared_groups = []

        for group in self.param_groups:
            params = []
            grads = []

            for p in group["params"]:
                grad: Tensor = p.grad

                if grad is None:
                    continue

                if grad.is_sparse:
                    raise RuntimeError(
                        "RADAR does not support sparse gradients."
                    )

                dtype = p.dtype

                if dtype.is_complex:
                    raise RuntimeError(
                        "RADAR does not currently support complex parameters."
                    )

                if dtype == torch.float16:
                    raise RuntimeError(
                        "RADAR does not currently support float16 parameters; "
                        "use AMP with float32 parameters."
                    )

                params.append(p)
                grads.append(grad)

            if params:
                prepared_groups.append((group, params, grads))

        foreach_available = None
        state_by_parameter = self.state
        migrate_legacy_state = self._migrate_legacy_param_state_
        step_to_int = self._step_to_int

        for group, params, grads in prepared_groups:
            beta1, beta2 = group["betas"]
            gamma = group["gamma"]

            exp_avgs = []
            exp_avg_sqs = []
            steps = []

            uniform_step = True
            first_step = None

            for p in params:
                state = state_by_parameter[p]

                # ----------------------------------------------------
                # Lazy initialization
                # ----------------------------------------------------
                if not state:
                    state["step"] = 0

                    state["exp_avg"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

                else:
                    # Safety for states loaded without __setstate__ migration.
                    if "prev_grad" in state:
                        migrate_legacy_state(
                            state,
                            beta1,
                            gamma,
                        )

                step = state["step"]
                if not isinstance(step, int):
                    step = step_to_int(step)

                step += 1
                state["step"] = step

                if first_step is None:
                    first_step = step
                elif step != first_step:
                    uniform_step = False

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                steps.append(step)

            kwargs = dict(
                lr=group["lr"],
                beta1=beta1,
                beta2=beta2,
                gamma=gamma,
                l=group["l"],
                delta=group["delta"],
                zeta=group["zeta"],
                weight_decay=group["weight_decay"],
                decoupled_weight_decay=group[
                    "decoupled_weight_decay"
                ],
            )

            use_foreach = group["foreach"]
            if use_foreach:
                if foreach_available is None:
                    foreach_available = self._can_use_foreach()
                use_foreach = foreach_available

            if use_foreach:
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
                    uniform_step,
                    first_step,
                    **kwargs,
                )

        return loss
