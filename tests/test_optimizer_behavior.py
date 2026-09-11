import copy
import unittest
from unittest import mock

import torch
from torch.optim.optimizer import Optimizer

from rad.optim import (
    AdaBayes,
    Adam,
    AdamW,
    DLPF,
    KFAdam,
    NAdam,
    NAG,
    RAD,
    RADAR,
    RGD,
    SGD,
    SWATS,
)


class PublicOptimizerSmokeTests(unittest.TestCase):
    def test_all_public_optimizers_complete_two_steps(self):
        constructors = {
            "RAD": lambda parameter: RAD([parameter]),
            "RADAR": lambda parameter: RADAR([parameter]),
            "Adam": lambda parameter: Adam([parameter]),
            "SGD": lambda parameter: SGD([parameter], lr=1e-3),
            "DLPF": lambda parameter: DLPF([parameter], lr=1e-3, momentum=0.9),
            "RGD": lambda parameter: RGD([parameter], lr=1e-3, momentum=0.9),
            "NAG": lambda parameter: NAG([parameter], lr=1e-3, momentum=0.9),
            "NAdam": lambda parameter: NAdam([parameter]),
            "SWATS": lambda parameter: SWATS([parameter]),
            "AdamW": lambda parameter: AdamW([parameter]),
            "KFAdam": lambda parameter: KFAdam([parameter]),
            "AdaBayes": lambda parameter: AdaBayes([parameter]),
        }

        for name, create_optimizer in constructors.items():
            with self.subTest(optimizer=name):
                parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
                optimizer = create_optimizer(parameter)

                for gradient in (
                    torch.tensor([0.2, -0.4]),
                    torch.tensor([-0.1, 0.3]),
                ):
                    parameter.grad = gradient.clone()
                    optimizer.step()

                self.assertTrue(torch.isfinite(parameter).all())


class ZeroMomentumTests(unittest.TestCase):
    def _assert_matches_sgd(self, optimizer_class):
        initial = torch.tensor([1.5, -0.5], dtype=torch.float64)
        candidate_parameter = torch.nn.Parameter(initial.clone())
        sgd_parameter = torch.nn.Parameter(initial.clone())

        candidate = optimizer_class(
            [candidate_parameter],
            lr=0.05,
            momentum=0,
            weight_decay=0.1,
            output_info=True,
        )

        sgd = SGD(
            [sgd_parameter],
            lr=0.05,
            momentum=0,
            weight_decay=0.1,
            output_info=True,
        )

        for gradient in (
            torch.tensor([0.2, -0.4], dtype=torch.float64),
            torch.tensor([-0.1, 0.3], dtype=torch.float64),
        ):
            candidate_parameter.grad = gradient.clone()
            sgd_parameter.grad = gradient.clone()

            candidate_result = candidate.step()
            sgd_result = sgd.step()

            self.assertTrue(torch.equal(candidate_parameter, sgd_parameter))
            self.assertEqual(candidate_result, sgd_result)

        self.assertEqual(len(candidate.state), 0)

    def test_nag_with_zero_momentum_matches_sgd(self):
        self._assert_matches_sgd(NAG)

    def test_dlpf_with_zero_momentum_matches_sgd(self):
        self._assert_matches_sgd(DLPF)

    def test_negative_momentum_is_rejected(self):
        for optimizer_class in (NAG, DLPF):
            parameter = torch.nn.Parameter(torch.tensor([1.0]))

            with self.subTest(optimizer=optimizer_class.__name__):
                with self.assertRaises(ValueError):
                    optimizer_class(
                        [parameter],
                        lr=0.1,
                        momentum=-0.1,
                    )


class RADARTests(unittest.TestCase):
    def _assert_matches_original_formula(
        self,
        *,
        beta1,
        foreach,
        dtype,
        weight_decay=0.0,
        decoupled_weight_decay=True,
    ):
        initial = torch.tensor([1.2, -0.7, 2.5], dtype=dtype)
        parameter = torch.nn.Parameter(initial.clone())
        reference = initial.clone()

        lr = 1e-3
        beta2 = 0.999
        gamma = 0.01
        residual_step = 1e-5
        delta = 1.0
        zeta = 1e-16

        optimizer = RADAR(
            [parameter],
            lr=lr,
            betas=(beta1, beta2),
            gamma=gamma,
            l=residual_step,
            delta=delta,
            zeta=zeta,
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            foreach=foreach,
        )

        corrected_moment = torch.zeros_like(reference)
        ordinary_ema = torch.zeros_like(reference)
        second_moment = torch.zeros_like(reference)
        previous_gradient = torch.zeros_like(reference)
        step = 0

        gradients = (
            [0.2, -0.4, 0.1],
            None,
            [-0.1, 0.3, -0.2],
            [0.05, -0.2, 0.4],
        )

        for values in gradients:
            if values is None:
                parameter.grad = None
                optimizer.step()
                continue

            gradient = torch.tensor(values, dtype=dtype)
            parameter.grad = gradient.clone()
            optimizer.step()

            with torch.no_grad():
                if weight_decay != 0.0:
                    if decoupled_weight_decay:
                        reference.mul_(1.0 - lr * weight_decay)
                    else:
                        gradient = gradient.add(
                            reference,
                            alpha=weight_decay,
                        )

                step += 1
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                corrected_moment.mul_(beta1)
                corrected_moment.add_(
                    gradient,
                    alpha=1.0 - beta1 + gamma,
                )
                corrected_moment.add_(
                    previous_gradient,
                    alpha=-gamma,
                )

                ordinary_ema.lerp_(gradient, 1.0 - beta1)
                second_moment.mul_(beta2)
                second_moment.addcmul_(
                    gradient,
                    gradient,
                    value=1.0 - beta2,
                )

                inverse_denominator = 1.0 / torch.sqrt(
                    delta**2
                    * second_moment
                    / bias_correction2
                    + zeta
                )

                reference.addcmul_(
                    corrected_moment,
                    inverse_denominator,
                    value=-(lr - residual_step) / bias_correction1,
                )
                reference.addcmul_(
                    gradient,
                    inverse_denominator,
                    value=-residual_step,
                )
                previous_gradient.copy_(gradient)

        tolerance = 1e-6 if dtype == torch.float32 else 1e-12
        self.assertTrue(
            torch.allclose(
                parameter,
                reference,
                rtol=tolerance,
                atol=tolerance,
            )
        )

        state = optimizer.state[parameter]
        self.assertEqual(state["step"], step)
        self.assertTrue(
            torch.allclose(
                state["exp_avg"],
                ordinary_ema,
                rtol=tolerance,
                atol=tolerance,
            )
        )
        self.assertNotIn("prev_grad", state)
        self.assertNotIn("prev_grad_valid", state)

    def test_default_residual_step_size_stays_fixed(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = RADAR([parameter], lr=1e-3)

        self.assertAlmostEqual(
            optimizer.param_groups[0]["l"],
            1e-5,
        )

        self.assertEqual(
            optimizer.param_groups[0]["weight_decay"],
            0,
        )

        optimizer.param_groups[0]["lr"] = 1e-4

        self.assertAlmostEqual(
            optimizer.param_groups[0]["l"],
            1e-5,
        )

    def test_missing_weight_decay_state_uses_current_default(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = RADAR([parameter])

        state_dict = optimizer.state_dict()
        del state_dict["param_groups"][0]["weight_decay"]

        restored_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        restored_optimizer = RADAR([restored_parameter])

        restored_optimizer.load_state_dict(state_dict)

        self.assertEqual(
            restored_optimizer.param_groups[0]["weight_decay"],
            0,
        )

    def test_added_parameter_group_gets_its_own_fixed_residual_step_size(self):
        first_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        second_parameter = torch.nn.Parameter(torch.tensor([2.0]))

        optimizer = RADAR([first_parameter], lr=1e-3)

        optimizer.add_param_group(
            {
                "params": [second_parameter],
                "lr": 2e-3,
            }
        )

        self.assertAlmostEqual(
            optimizer.param_groups[0]["l"],
            1e-5,
        )

        self.assertAlmostEqual(
            optimizer.param_groups[1]["l"],
            2e-5,
        )

        first_parameter.grad = torch.tensor([0.1])
        second_parameter.grad = torch.tensor([0.2])

        optimizer.step()

        self.assertTrue(torch.isfinite(first_parameter).all())
        self.assertTrue(torch.isfinite(second_parameter).all())

    def test_first_step_matches_documented_update(self):
        parameter = torch.nn.Parameter(
            torch.tensor(
                [1.0, -2.0],
                dtype=torch.float64,
            )
        )

        gradient = torch.tensor(
            [0.2, -0.4],
            dtype=torch.float64,
        )

        parameter.grad = gradient.clone()
        initial = parameter.detach().clone()

        lr = 1e-3
        beta1, beta2 = 0.9, 0.999
        gamma = 0.01
        residual_step = 1e-5
        delta = 1.0
        zeta = 1e-16

        optimizer = RADAR(
            [parameter],
            lr=lr,
            betas=(beta1, beta2),
            gamma=gamma,
            l=residual_step,
            delta=delta,
            zeta=zeta,
            weight_decay=0,
        )

        exp_avg = (1 - beta1 + gamma) * gradient
        exp_avg_sq = (1 - beta2) * gradient.square()

        bias_correction1 = 1 - beta1
        bias_correction2 = 1 - beta2

        denominator = 1 / torch.sqrt(
            delta**2 * exp_avg_sq / bias_correction2 + zeta
        )

        expected = initial.clone()

        expected.addcmul_(
            exp_avg,
            denominator,
            value=-lr / bias_correction1,
        )

        expected.addcmul_(
            exp_avg,
            denominator,
            value=residual_step / bias_correction1,
        )

        expected.addcmul_(
            gradient,
            denominator,
            value=-residual_step,
        )

        optimizer.step()

        self.assertTrue(
            torch.allclose(
                parameter,
                expected,
                rtol=1e-12,
                atol=1e-12,
            )
        )

        state = optimizer.state[parameter]

        self.assertEqual(
            state["step"],
            1,
        )

        # New reparameterized RADAR stores the ordinary EMA
        # instead of storing prev_grad.
        expected_exp_avg = (1 - beta1) * gradient

        self.assertTrue(
            torch.allclose(
                state["exp_avg"],
                expected_exp_avg,
                rtol=1e-12,
                atol=1e-12,
            )
        )

        self.assertNotIn(
            "prev_grad",
            state,
        )

    def test_reparameterization_matches_original_formula(self):
        for beta1, dtype in (
            (0.0, torch.float64),
            (1e-12, torch.float32),
            (0.9, torch.float64),
        ):
            for foreach in (False, True):
                with self.subTest(
                    beta1=beta1,
                    dtype=dtype,
                    foreach=foreach,
                ):
                    self._assert_matches_original_formula(
                        beta1=beta1,
                        foreach=foreach,
                        dtype=dtype,
                    )

    def test_weight_decay_matches_original_formula(self):
        for decoupled in (False, True):
            for foreach in (False, True):
                with self.subTest(
                    decoupled=decoupled,
                    foreach=foreach,
                ):
                    self._assert_matches_original_formula(
                        beta1=0.8,
                        foreach=foreach,
                        dtype=torch.float64,
                        weight_decay=0.2,
                        decoupled_weight_decay=decoupled,
                    )

    def test_foreach_matches_single_with_different_active_steps(self):
        initial_values = (
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([-1.0, 0.5], dtype=torch.float64),
            torch.tensor([0.25, -0.75], dtype=torch.float32),
        )
        single_parameters = [
            torch.nn.Parameter(value.clone())
            for value in initial_values
        ]
        foreach_parameters = [
            torch.nn.Parameter(value.clone())
            for value in initial_values
        ]

        single = RADAR(single_parameters, foreach=False)
        foreach = RADAR(foreach_parameters, foreach=True)

        gradient_pairs = (
            ([0.2, -0.1], [0.4, 0.3], [-0.3, 0.2]),
            ([0.1, 0.5], None, None),
            ([-0.2, 0.6], None, [0.3, -0.4]),
            ([0.7, -0.4], [0.2, -0.3], [0.1, 0.5]),
        )

        for pair in gradient_pairs:
            for parameter, values in zip(single_parameters, pair):
                parameter.grad = (
                    None
                    if values is None
                    else torch.tensor(values, dtype=parameter.dtype)
                )
            for parameter, values in zip(foreach_parameters, pair):
                parameter.grad = (
                    None
                    if values is None
                    else torch.tensor(values, dtype=parameter.dtype)
                )

            single.step()
            foreach.step()

        for single_parameter, foreach_parameter in zip(
            single_parameters,
            foreach_parameters,
        ):
            self.assertTrue(
                torch.equal(single_parameter, foreach_parameter)
            )
            single_state = single.state[single_parameter]
            foreach_state = foreach.state[foreach_parameter]
            self.assertEqual(single_state["step"], foreach_state["step"])
            self.assertTrue(
                torch.equal(
                    single_state["exp_avg"],
                    foreach_state["exp_avg"],
                )
            )
            self.assertTrue(
                torch.equal(
                    single_state["exp_avg_sq"],
                    foreach_state["exp_avg_sq"],
                )
            )

        self.assertEqual(
            [single.state[parameter]["step"] for parameter in single_parameters],
            [4, 2, 3],
        )

    def test_missing_foreach_capability_falls_back_to_single_tensor(self):
        initial = torch.tensor([1.0, -2.0], dtype=torch.float64)
        fallback_parameter = torch.nn.Parameter(initial.clone())
        single_parameter = torch.nn.Parameter(initial.clone())
        fallback = RADAR([fallback_parameter])
        single = RADAR([single_parameter], foreach=False)
        gradient = torch.tensor([0.2, -0.4], dtype=torch.float64)

        fallback_parameter.grad = gradient.clone()
        single_parameter.grad = gradient.clone()

        with mock.patch.object(
            Optimizer,
            "_group_tensors_by_device_and_dtype",
            None,
            create=True,
        ):
            fallback.step()

        single.step()
        self.assertTrue(torch.equal(fallback_parameter, single_parameter))

    def test_old_checkpoint_respects_explicit_foreach_preference(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        source = RADAR([parameter])
        state_dict = source.state_dict()
        del state_dict["param_groups"][0]["foreach"]

        restored_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        restored = RADAR([restored_parameter], foreach=False)
        restored.load_state_dict(state_dict)

        self.assertFalse(restored.param_groups[0]["foreach"])

    def test_legacy_corrected_momentum_is_migrated(self):
        ordinary_ema = torch.tensor([0.2, -0.3], dtype=torch.float64)
        previous_gradient = torch.tensor(
            [-0.1, 0.4],
            dtype=torch.float64,
        )
        migration_cases = (
            (0.9, 0.01, ordinary_ema),
            (0.9, 0.0, ordinary_ema),
            (0.0, 0.01, previous_gradient),
            (0.2, 0.2, torch.zeros_like(ordinary_ema)),
        )

        for beta1, gamma, expected_ema in migration_cases:
            with self.subTest(beta1=beta1, gamma=gamma):
                if beta1 == 0.0 or beta1 == gamma:
                    corrected_moment = previous_gradient.clone()
                else:
                    corrected_moment = (
                        (beta1 - gamma) / beta1 * ordinary_ema
                        + gamma / beta1 * previous_gradient
                    )

                parameter = torch.nn.Parameter(
                    torch.tensor([1.0, -2.0], dtype=torch.float64)
                )
                optimizer = RADAR(
                    [parameter],
                    betas=(beta1, 0.999),
                    gamma=gamma,
                    foreach=False,
                )
                state_dict = optimizer.state_dict()
                legacy_second_moment = torch.tensor(
                    [0.03, 0.05],
                    dtype=torch.float64,
                )
                state_dict["state"][0] = {
                    "step": torch.tensor(3),
                    "exp_avg": corrected_moment.clone(),
                    "exp_avg_sq": legacy_second_moment.clone(),
                    "prev_grad": previous_gradient.clone(),
                    "prev_grad_valid": False,
                }

                optimizer.load_state_dict(state_dict)
                state = optimizer.state[parameter]

                self.assertEqual(state["step"], 3)
                self.assertTrue(
                    torch.allclose(
                        state["exp_avg"],
                        expected_ema,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                )
                self.assertNotIn("prev_grad", state)
                self.assertNotIn("prev_grad_valid", state)

                reference = parameter.detach().clone()
                reference_moment = corrected_moment.clone()
                reference_second_moment = legacy_second_moment.clone()
                reference_previous_gradient = previous_gradient.clone()
                reference_step = 3

                for values in ([0.3, -0.2], None, [-0.4, 0.1]):
                    if values is None:
                        parameter.grad = None
                        optimizer.step()
                        continue

                    gradient = torch.tensor(values, dtype=torch.float64)
                    parameter.grad = gradient.clone()
                    optimizer.step()

                    with torch.no_grad():
                        reference_step += 1
                        bias_correction1 = 1.0 - beta1**reference_step
                        bias_correction2 = 1.0 - 0.999**reference_step

                        reference_moment.mul_(beta1)
                        reference_moment.add_(
                            gradient,
                            alpha=1.0 - beta1 + gamma,
                        )
                        reference_moment.add_(
                            reference_previous_gradient,
                            alpha=-gamma,
                        )
                        reference_second_moment.mul_(0.999)
                        reference_second_moment.addcmul_(
                            gradient,
                            gradient,
                            value=0.001,
                        )

                        inverse_denominator = 1.0 / torch.sqrt(
                            reference_second_moment
                            / bias_correction2
                            + 1e-16
                        )
                        reference.addcmul_(
                            reference_moment,
                            inverse_denominator,
                            value=-(1e-3 - 1e-5)
                            / bias_correction1,
                        )
                        reference.addcmul_(
                            gradient,
                            inverse_denominator,
                            value=-1e-5,
                        )
                        reference_previous_gradient.copy_(gradient)

                self.assertTrue(
                    torch.allclose(
                        parameter,
                        reference,
                        rtol=1e-12,
                        atol=1e-12,
                    )
                )

    def test_state_dict_roundtrip_continues_identically(self):
        for beta1 in (0.0, 0.9):
            for foreach in (False, True):
                with self.subTest(beta1=beta1, foreach=foreach):
                    parameter = torch.nn.Parameter(
                        torch.tensor([1.0, -2.0], dtype=torch.float64)
                    )
                    optimizer = RADAR(
                        [parameter],
                        betas=(beta1, 0.999),
                        foreach=foreach,
                    )

                    for values in ([0.2, -0.4], [-0.1, 0.3]):
                        parameter.grad = torch.tensor(
                            values,
                            dtype=torch.float64,
                        )
                        optimizer.step()

                    state_dict = copy.deepcopy(optimizer.state_dict())
                    restored_parameter = torch.nn.Parameter(
                        parameter.detach().clone()
                    )
                    restored = RADAR(
                        [restored_parameter],
                        betas=(beta1, 0.999),
                        foreach=foreach,
                    )
                    restored.load_state_dict(state_dict)

                    for values in (None, [0.05, -0.2], [0.4, 0.1]):
                        parameter.grad = (
                            None
                            if values is None
                            else torch.tensor(values, dtype=torch.float64)
                        )
                        restored_parameter.grad = (
                            None
                            if values is None
                            else torch.tensor(values, dtype=torch.float64)
                        )
                        optimizer.step()
                        restored.step()

                    self.assertTrue(
                        torch.equal(parameter, restored_parameter)
                    )
                    original_state = optimizer.state[parameter]
                    restored_state = restored.state[restored_parameter]
                    self.assertEqual(
                        original_state["step"],
                        restored_state["step"],
                    )
                    self.assertTrue(
                        torch.equal(
                            original_state["exp_avg"],
                            restored_state["exp_avg"],
                        )
                    )
                    self.assertTrue(
                        torch.equal(
                            original_state["exp_avg_sq"],
                            restored_state["exp_avg_sq"],
                        )
                    )

    def test_invalid_legacy_checkpoint_has_clear_error(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = RADAR([parameter], foreach=False)
        state_dict = optimizer.state_dict()
        state_dict["state"][0] = {
            "step": 1,
            "exp_avg_sq": torch.tensor([0.01]),
            "prev_grad": torch.tensor([0.2]),
        }

        with self.assertRaisesRegex(
            RuntimeError,
            "contains prev_grad but no exp_avg",
        ):
            optimizer.load_state_dict(state_dict)

    def test_float16_parameters_are_rejected_before_mutation(self):
        for foreach in (False, True):
            for gradient_value in (0.0, 1e-3):
                with self.subTest(
                    foreach=foreach,
                    gradient=gradient_value,
                ):
                    parameter = torch.nn.Parameter(
                        torch.tensor([1.0, -2.0], dtype=torch.float16)
                    )
                    initial = parameter.detach().clone()
                    optimizer = RADAR([parameter], foreach=foreach)
                    parameter.grad = torch.full_like(
                        parameter,
                        gradient_value,
                    )

                    with self.assertRaisesRegex(
                        RuntimeError,
                        "does not currently support float16",
                    ):
                        optimizer.step()

                    self.assertTrue(torch.equal(parameter, initial))
                    self.assertEqual(len(optimizer.state), 0)

        fp32_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        fp16_parameter = torch.nn.Parameter(
            torch.tensor([2.0], dtype=torch.float16)
        )
        optimizer = RADAR(
            [{"params": [fp32_parameter]}, {"params": [fp16_parameter]}]
        )
        fp32_parameter.grad = torch.tensor([0.2])
        fp16_parameter.grad = torch.tensor([0.1], dtype=torch.float16)

        with self.assertRaisesRegex(
            RuntimeError,
            "does not currently support float16",
        ):
            optimizer.step()

        self.assertTrue(torch.equal(fp32_parameter, torch.tensor([1.0])))
        self.assertTrue(
            torch.equal(
                fp16_parameter,
                torch.tensor([2.0], dtype=torch.float16),
            )
        )
        self.assertEqual(len(optimizer.state), 0)

    def test_unsupported_parameters_are_rejected_before_mutation(self):
        invalid_cases = (
            (
                "sparse",
                torch.nn.Parameter(torch.tensor([2.0, -1.0])),
                torch.sparse_coo_tensor(
                    torch.tensor([[0]]),
                    torch.tensor([0.1]),
                    (2,),
                ),
                "does not support sparse gradients",
            ),
            (
                "complex",
                torch.nn.Parameter(torch.tensor([2.0 + 1.0j])),
                torch.tensor([0.1 - 0.2j]),
                "does not currently support complex parameters",
            ),
        )

        for name, invalid_parameter, invalid_gradient, message in invalid_cases:
            with self.subTest(case=name):
                valid_parameter = torch.nn.Parameter(torch.tensor([1.0]))
                valid_initial = valid_parameter.detach().clone()
                invalid_initial = invalid_parameter.detach().clone()
                optimizer = RADAR(
                    [
                        {"params": [valid_parameter]},
                        {"params": [invalid_parameter]},
                    ]
                )
                valid_parameter.grad = torch.tensor([0.2])
                invalid_parameter.grad = invalid_gradient

                with self.assertRaisesRegex(RuntimeError, message):
                    optimizer.step()

                self.assertTrue(
                    torch.equal(valid_parameter, valid_initial)
                )
                self.assertTrue(
                    torch.equal(invalid_parameter, invalid_initial)
                )
                self.assertEqual(len(optimizer.state), 0)

    def test_bfloat16_parameters_remain_finite(self):
        for foreach in (False, True):
            with self.subTest(foreach=foreach):
                parameter = torch.nn.Parameter(
                    torch.tensor([1.0, -2.0], dtype=torch.bfloat16)
                )
                optimizer = RADAR([parameter], foreach=foreach)

                for gradient_value in (0.0, 1e-3, 0.2):
                    parameter.grad = torch.full_like(
                        parameter,
                        gradient_value,
                    )
                    optimizer.step()

                self.assertTrue(torch.isfinite(parameter).all())


if __name__ == "__main__":
    unittest.main()
