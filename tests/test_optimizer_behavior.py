import unittest

import torch

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
                    optimizer_class([parameter], lr=0.1, momentum=-0.1)


class RADARTests(unittest.TestCase):
    def test_default_residual_step_size_stays_fixed(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = RADAR([parameter], lr=1e-3)

        self.assertAlmostEqual(optimizer.param_groups[0]["l"], 1e-5)
        self.assertEqual(optimizer.param_groups[0]["weight_decay"], 0)
        optimizer.param_groups[0]["lr"] = 1e-4
        self.assertAlmostEqual(optimizer.param_groups[0]["l"], 1e-5)

    def test_missing_weight_decay_state_uses_current_default(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = RADAR([parameter])
        state_dict = optimizer.state_dict()
        del state_dict["param_groups"][0]["weight_decay"]

        restored_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        restored_optimizer = RADAR([restored_parameter])
        restored_optimizer.load_state_dict(state_dict)

        self.assertEqual(restored_optimizer.param_groups[0]["weight_decay"], 0)

    def test_added_parameter_group_gets_its_own_fixed_residual_step_size(self):
        first_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        second_parameter = torch.nn.Parameter(torch.tensor([2.0]))
        optimizer = RADAR([first_parameter], lr=1e-3)

        optimizer.add_param_group({"params": [second_parameter], "lr": 2e-3})

        self.assertAlmostEqual(optimizer.param_groups[0]["l"], 1e-5)
        self.assertAlmostEqual(optimizer.param_groups[1]["l"], 2e-5)
        first_parameter.grad = torch.tensor([0.1])
        second_parameter.grad = torch.tensor([0.2])
        optimizer.step()
        self.assertTrue(torch.isfinite(first_parameter).all())
        self.assertTrue(torch.isfinite(second_parameter).all())

    def test_first_step_matches_documented_update(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
        gradient = torch.tensor([0.2, -0.4], dtype=torch.float64)
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
        denominator = 1 / torch.sqrt(delta**2 * exp_avg_sq / bias_correction2 + zeta)
        expected = initial.clone()
        expected.addcmul_(exp_avg, denominator, value=-lr / bias_correction1)
        expected.addcmul_(exp_avg, denominator, value=residual_step / bias_correction1)
        expected.addcmul_(gradient, denominator, value=-residual_step)

        optimizer.step()

        self.assertTrue(torch.allclose(parameter, expected, rtol=1e-12, atol=1e-12))
        state = optimizer.state[parameter]
        self.assertEqual(state["step"], 1)
        self.assertTrue(torch.equal(state["prev_grad"], gradient))


if __name__ == "__main__":
    unittest.main()
