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
            Gradient residual correction coefficient.
            Default: 0.01.

        l (float, optional):
            Residual correction step size.

            If l is None, the optimizer initializes

                l = 0.01 * initial_lr

            for each parameter group. The resulting value remains
            fixed when a learning-rate scheduler changes lr.

        delta (float, optional):
            Scaling coefficient in the adaptive preconditioner.
            Default: 1.

        zeta (float, optional):
            Numerical stability coefficient.
            Default: 1e-16.

        weight_decay (float, optional):
            Weight decay coefficient.
            Default: 0.

        decoupled_weight_decay (bool, optional):
            Whether to use AdamW-style decoupled weight decay.
            Default: True.

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
            raise ValueError(
                f"Invalid learning rate: {lr}"
            )

        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 0: {betas[0]}"
            )

        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                f"Invalid beta parameter at index 1: {betas[1]}"
            )

        if gamma < 0.0:
            raise ValueError(
                f"Invalid gamma value: {gamma}"
            )

        if l is not None and l < 0.0:
            raise ValueError(
                f"Invalid l value: {l}"
            )

        if delta <= 0.0:
            raise ValueError(
                f"Invalid delta value: {delta}"
            )

        if zeta <= 0.0:
            raise ValueError(
                f"Invalid zeta value: {zeta}"
            )

        if weight_decay < 0.0:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}"
            )

        if not isinstance(foreach, bool):
            raise TypeError(
                f"foreach must be bool, got {type(foreach)}"
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
            foreach=foreach,
        )

        super().__init__(
            params,
            defaults,
        )

    def add_param_group(self, param_group):
        """Add a parameter group and initialize its fixed residual step size."""
        super().add_param_group(param_group)

        group = self.param_groups[-1]
        if group["l"] is None:
            group["l"] = 0.01 * group["lr"]

    # ================================================================
    # Utility
    # ================================================================

    @staticmethod
    def _step_to_int(step):
        """Convert an old checkpoint step representation to Python int."""

        if torch.is_tensor(step):
            if step.numel() != 1:
                raise RuntimeError(
                    "RADAR state['step'] Tensor must contain one element."
                )

            return int(
                step.detach().cpu().item()
            )

        return int(step)

    # ================================================================
    # State loading
    # ================================================================

    def __setstate__(self, state):
        super().__setstate__(state)

        for group in self.param_groups:

            group.setdefault(
                "gamma",
                0.01,
            )


            group.setdefault(
                "l",
                None,
            )

            if group["l"] is None:
                group["l"] = 0.01 * group["lr"]

            group.setdefault(
                "delta",
                1.0,
            )

            group.setdefault(
                "zeta",
                1e-16,
            )

            group.setdefault(
                "weight_decay",
                0.0,
            )

            group.setdefault(
                "decoupled_weight_decay",
                True,
            )

            if group.get("foreach") is None:
                group["foreach"] = True

            for p in group["params"]:

                p_state = self.state.get(
                    p,
                    None,
                )

                if not p_state:
                    continue

                if "step" in p_state:
                    p_state["step"] = (
                        self._step_to_int(
                            p_state["step"]
                        )
                    )

                p_state.setdefault(
                    "prev_grad_valid",
                    bool(
                        p_state.get(
                            "step",
                            0,
                        ) > 0
                        and "prev_grad" in p_state
                    ),
                )

    # ================================================================
    # Single-tensor implementation
    # ================================================================

    @staticmethod
    def _single_tensor_radar(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        prev_grads,
        prev_grad_valids,
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
        delta_sq = (
            delta * delta
        )

        for (
            p,
            grad,
            exp_avg,
            exp_avg_sq,
            prev_grad,
            prev_grad_valid,
            step,
        ) in zip(
            params,
            grads,
            exp_avgs,
            exp_avg_sqs,
            prev_grads,
            prev_grad_valids,
            steps,
        ):

            # ========================================================
            # Weight decay
            # ========================================================

            if weight_decay != 0:

                # ----------------------------------------------------
                # Decoupled weight decay:
                #
                # theta <- (1 - lr * wd) * theta
                # ----------------------------------------------------

                if decoupled_weight_decay:

                    p.mul_(
                        1.0
                        - lr
                        * weight_decay
                    )

                # ----------------------------------------------------
                # Coupled L2 weight decay:
                #
                # g <- g + wd * theta
                # ----------------------------------------------------

                else:

                    grad = grad.add(
                        p,
                        alpha=weight_decay,
                    )

            # ========================================================
            # Bias correction
            # ========================================================

            bias_correction1 = (
                1.0
                - beta1 ** step
            )

            bias_correction2 = (
                1.0
                - beta2 ** step
            )

            # ========================================================
            # First moment
            #
            # Base EMA:
            #
            # m_t =
            # beta1 * m_{t-1}
            # + (1-beta1) * g_t
            #
            # lerp implements:
            #
            # m <- m + (g-m) * (1-beta1)
            #
            #    = beta1*m + (1-beta1)*g
            # ========================================================

            exp_avg.lerp_(
                grad,
                1.0 - beta1,
            )

            # ========================================================
            # RADAR residual correction
            #
            # Initial step:
            #
            #     prev_grad = 0
            #     prev_grad_valid = True
            #
            # therefore:
            #
            #     gamma * (g_1 - 0)
            #
            # is applied exactly as in the original RADAR.
            #
            # After grad=None, prev_grad_valid becomes False and the
            # first recovered gradient skips this residual correction.
            # ========================================================

            if (
                gamma != 0.0
                and prev_grad_valid
            ):

                exp_avg.add_(
                    grad,
                    alpha=gamma,
                )

                exp_avg.add_(
                    prev_grad,
                    alpha=-gamma,
                )

            # ========================================================
            # Second moment
            #
            # v_t =
            #
            # beta2 * v_{t-1}
            # + (1-beta2) * g_t^2
            # ========================================================

            exp_avg_sq.mul_(
                beta2
            )

            exp_avg_sq.addcmul_(
                grad,
                grad,
                value=(
                    1.0 - beta2
                ),
            )

            # ========================================================
            # Optimized denominator
            #
            # Original:
            #
            # D_t =
            #
            # 1 /
            # sqrt(
            #     delta^2 * v_t / bc2
            #     + zeta
            # )
            #
            #
            # Algebraically:
            #
            # D_t =
            #
            # sqrt(bc2) / delta
            # -----------------------------------
            # sqrt(
            #     v_t
            #     + zeta * bc2 / delta^2
            # )
            #
            # ========================================================

            denominator = (
                exp_avg_sq.clone(
                    memory_format=(
                        torch.preserve_format
                    )
                )
            )

            denominator.add_(
                zeta
                * bias_correction2
                / delta_sq
            )

            denominator.sqrt_()

            preconditioner_scale = (
                bias_correction2 ** 0.5
                / delta
            )

            # ========================================================
            # Combined RADAR update
            #
            # Original:
            #
            # - lr/bc1 * m * D
            # + l /bc1 * m * D
            # - l       * g * D
            #
            # =
            #
            # -(lr-l)/bc1 * m * D
            # - l           * g * D
            #
            # ========================================================

            momentum_step_size = (
                -(
                    lr - l
                )
                / bias_correction1
                * preconditioner_scale
            )

            gradient_step_size = (
                -l
                * preconditioner_scale
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

            # ========================================================
            # Save current effective gradient
            # ========================================================

            prev_grad.copy_(
                grad
            )

    # ================================================================
    # Foreach implementation for one device/dtype group
    # ================================================================

    @staticmethod
    def _foreach_bucket(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        prev_grads,
        indices,
        prev_grad_valids,
        steps,
        all_prev_grad_valid,
        no_prev_grad_valid,
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

        # ============================================================
        # Weight decay
        # ============================================================

        if weight_decay != 0:

            # --------------------------------------------------------
            # Decoupled decay.
            # --------------------------------------------------------

            if decoupled_weight_decay:

                torch._foreach_mul_(
                    params,
                    (
                        1.0
                        - lr
                        * weight_decay
                    ),
                )

            # --------------------------------------------------------
            # Coupled weight decay.
            # --------------------------------------------------------

            else:

                grads = list(
                    torch._foreach_add(
                        grads,
                        params,
                        alpha=weight_decay,
                    )
                )

        # ============================================================
        # Base first-moment EMA
        #
        # m =
        #
        # beta1*m
        # + (1-beta1)*g
        #
        # One foreach kernel instead of mul + add.
        # ============================================================

        torch._foreach_lerp_(
            exp_avgs,
            grads,
            1.0 - beta1,
        )

        # ============================================================
        # Residual correction
        # ============================================================

        if (
            gamma != 0.0
            and not no_prev_grad_valid
        ):


            if all_prev_grad_valid:

                torch._foreach_add_(
                    exp_avgs,
                    grads,
                    alpha=gamma,
                )

                torch._foreach_add_(
                    exp_avgs,
                    prev_grads,
                    alpha=-gamma,
                )

            # --------------------------------------------------------
            # Mixed case:
            # --------------------------------------------------------

            else:

                valid_exp_avgs = []
                valid_grads = []
                valid_prev_grads = []

                for local_index, global_index in enumerate(
                    indices
                ):

                    if prev_grad_valids[
                        global_index
                    ]:

                        valid_exp_avgs.append(
                            exp_avgs[
                                local_index
                            ]
                        )

                        valid_grads.append(
                            grads[
                                local_index
                            ]
                        )

                        valid_prev_grads.append(
                            prev_grads[
                                local_index
                            ]
                        )

                if len(valid_exp_avgs) != 0:

                    torch._foreach_add_(
                        valid_exp_avgs,
                        valid_grads,
                        alpha=gamma,
                    )

                    torch._foreach_add_(
                        valid_exp_avgs,
                        valid_prev_grads,
                        alpha=-gamma,
                    )

        # ============================================================
        # Second moment
        # ============================================================

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

        delta_sq = (
            delta * delta
        )

        # ============================================================
        # Bias correction
        # ============================================================

        if uniform_step:

            bias_correction1 = (
                1.0
                - beta1
                ** uniform_step_value
            )

            bias_correction2 = (
                1.0
                - beta2
                ** uniform_step_value
            )

            # ========================================================
            # denominator =
            #
            # sqrt(
            #     v_t
            #     + zeta * bc2 / delta^2
            # )
            # ========================================================

            denominator_shift = (
                zeta
                * bias_correction2
                / delta_sq
            )

            denominators = list(
                torch._foreach_add(
                    exp_avg_sqs,
                    denominator_shift,
                )
            )

            torch._foreach_sqrt_(
                denominators
            )

            preconditioner_scale = (
                bias_correction2 ** 0.5
                / delta
            )

            momentum_step_size = (
                -(
                    lr - l
                )
                / bias_correction1
                * preconditioner_scale
            )

            gradient_step_size = (
                -l
                * preconditioner_scale
            )

            # ========================================================
            # Two final multi-tensor updates
            # ========================================================

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

        # ============================================================
        # Rare path:
        #
        # parameters have different update counters because some
        # parameters had grad=None.
        # ============================================================

        else:

            device_steps = [
                steps[index]
                for index in indices
            ]

            bias_correction1 = [
                (
                    1.0
                    - beta1 ** step
                )
                for step
                in device_steps
            ]

            bias_correction2 = [
                (
                    1.0
                    - beta2 ** step
                )
                for step
                in device_steps
            ]

            denominator_shifts = [
                (
                    zeta
                    * bc2
                    / delta_sq
                )
                for bc2
                in bias_correction2
            ]

            denominators = list(
                torch._foreach_add(
                    exp_avg_sqs,
                    denominator_shifts,
                )
            )

            torch._foreach_sqrt_(
                denominators
            )

            preconditioner_scales = [
                (
                    bc2 ** 0.5
                    / delta
                )
                for bc2
                in bias_correction2
            ]

            momentum_step_sizes = [
                (
                    -(
                        lr - l
                    )
                    / bc1
                    * scale
                )
                for bc1, scale
                in zip(
                    bias_correction1,
                    preconditioner_scales,
                )
            ]

            gradient_step_sizes = [
                (
                    -l
                    * scale
                )
                for scale
                in preconditioner_scales
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

        # Release the large temporary TensorList as soon as possible.
        del denominators

        # ============================================================
        # Save current effective gradients
        # ============================================================

        torch._foreach_copy_(
            prev_grads,
            grads,
        )

    # ================================================================
    # Multi-tensor foreach dispatch
    # ================================================================

    @classmethod
    def _multi_tensor_radar(
        cls,
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        prev_grads,
        prev_grad_valids,
        steps,
        valid_prev_grad_count,
        uniform_step,
        uniform_step_value,
        **kwargs,
    ):

        grouped_tensors = (
            Optimizer._group_tensors_by_device_and_dtype(
                [
                    params,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    prev_grads,
                ],
                with_indices=True,
            )
        )

        all_prev_grad_valid = (
            valid_prev_grad_count
            == len(params)
        )

        no_prev_grad_valid = (
            valid_prev_grad_count
            == 0
        )

        for (
            device_tensor_lists,
            indices,
        ) in grouped_tensors.values():

            (
                device_params,
                device_grads,
                device_exp_avgs,
                device_exp_avg_sqs,
                device_prev_grads,
            ) = device_tensor_lists

            cls._foreach_bucket(
                device_params,
                device_grads,
                device_exp_avgs,
                device_exp_avg_sqs,
                device_prev_grads,
                indices,
                prev_grad_valids,
                steps,
                all_prev_grad_valid,
                no_prev_grad_valid,
                uniform_step,
                uniform_step_value,
                **kwargs,
            )

    # ================================================================
    # Main optimizer step
    # ================================================================

    @torch.no_grad()
    def step(
        self,
        closure=None,
    ):
        loss = None

        if closure is not None:

            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:

            params = []
            grads = []

            exp_avgs = []
            exp_avg_sqs = []

            prev_grads = []
            prev_grad_valids = []

            steps = []

            # --------------------------------------------------------
            # Metadata accumulated WHILE gathering parameters.
            # --------------------------------------------------------

            valid_prev_grad_count = 0

            uniform_step = True
            first_step = None

            # ========================================================
            # Gather active parameters
            # ========================================================

            for p in group["params"]:

                grad: Tensor = p.grad

                # ====================================================
                # Missing gradient
                # ====================================================

                if grad is None:

                    state = self.state.get(
                        p,
                        None,
                    )

                    if state:


                        state[
                            "prev_grad_valid"
                        ] = False

                    continue

                # ====================================================
                # Unsupported gradients / parameters
                # ====================================================

                if grad.is_sparse:

                    raise RuntimeError(
                        "RADAR does not support sparse gradients."
                    )

                if torch.is_complex(p):

                    raise RuntimeError(
                        "RADAR does not currently support "
                        "complex parameters."
                    )

                state = self.state[p]

                # ====================================================
                # Lazy initialization
                # ====================================================

                if len(state) == 0:

                    # Python int:
                    #
                    # no Tensor step and no GPU synchronization.
                    state["step"] = 0

                    state["exp_avg"] = (
                        torch.zeros_like(
                            p,
                            memory_format=(
                                torch.preserve_format
                            ),
                        )
                    )

                    state["exp_avg_sq"] = (
                        torch.zeros_like(
                            p,
                            memory_format=(
                                torch.preserve_format
                            ),
                        )
                    )

                    state["prev_grad"] = (
                        torch.zeros_like(
                            p,
                            memory_format=(
                                torch.preserve_format
                            ),
                        )
                    )

                    state[
                        "prev_grad_valid"
                    ] = True

                # ====================================================
                # Read OLD residual-valid flag before changing it.
                # ====================================================

                prev_grad_valid = bool(
                    state[
                        "prev_grad_valid"
                    ]
                )

                if prev_grad_valid:
                    valid_prev_grad_count += 1

                # ====================================================
                # Increment Python integer step
                # ====================================================

                state["step"] += 1

                step = state["step"]


                if first_step is None:

                    first_step = step

                elif step != first_step:

                    uniform_step = False

                # ====================================================
                # Build tensor lists
                # ====================================================

                params.append(
                    p
                )

                grads.append(
                    grad
                )

                exp_avgs.append(
                    state["exp_avg"]
                )

                exp_avg_sqs.append(
                    state["exp_avg_sq"]
                )

                prev_grads.append(
                    state["prev_grad"]
                )

                prev_grad_valids.append(
                    prev_grad_valid
                )

                steps.append(
                    step
                )


                state[
                    "prev_grad_valid"
                ] = True

            if len(params) == 0:
                continue

            # ========================================================
            # Hyperparameters
            # ========================================================

            lr = group["lr"]
            l = group["l"]

            beta1, beta2 = (
                group["betas"]
            )

            kwargs = dict(
                lr=lr,
                beta1=beta1,
                beta2=beta2,
                gamma=group["gamma"],
                l=l,
                delta=group["delta"],
                zeta=group["zeta"],
                weight_decay=group[
                    "weight_decay"
                ],
                decoupled_weight_decay=group[
                    "decoupled_weight_decay"
                ],
            )

            if group["foreach"]:

                self._multi_tensor_radar(
                    params,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    prev_grads,
                    prev_grad_valids,
                    steps,
                    valid_prev_grad_count,
                    uniform_step,
                    first_step,
                    **kwargs,
                )

            else:

                self._single_tensor_radar(
                    params,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    prev_grads,
                    prev_grad_valids,
                    steps,
                    **kwargs,
                )

        return loss
