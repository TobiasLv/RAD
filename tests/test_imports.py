import unittest

import rad
import rad.optim as optim
from torch.optim import Optimizer


PUBLIC_OPTIMIZERS = (
    "RAD",
    "RADAR",
    "Adam",
    "SGD",
    "DLPF",
    "RGD",
    "NAG",
    "NAdam",
    "SWATS",
    "AdamW",
    "KFAdam",
    "AdaBayes",
)


class ImportTests(unittest.TestCase):
    def test_rad_package_imports(self):
        self.assertIsNotNone(rad)
        self.assertTrue(issubclass(optim.RAD, Optimizer))

    def test_documented_optimizers_are_exported(self):
        self.assertEqual(tuple(optim.__all__), PUBLIC_OPTIMIZERS)
        for name in PUBLIC_OPTIMIZERS:
            with self.subTest(optimizer=name):
                self.assertTrue(issubclass(getattr(optim, name), Optimizer))

    def test_documented_from_imports(self):
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

        imported = (RAD, RADAR, Adam, SGD, DLPF, RGD, NAG, NAdam, SWATS, AdamW, KFAdam, AdaBayes)
        self.assertEqual(tuple(cls.__name__ for cls in imported), PUBLIC_OPTIMIZERS)


if __name__ == "__main__":
    unittest.main()
