import unittest
from check.numerical import *
from check.approach_great_times import *
from check.common import *
import warnings
import torch
import numpy as np


warnings.filterwarnings("ignore")

class TestNumerical(unittest.TestCase):
    def test_TT_wide(self):
        numerical_result = TT_wide(
            si, li, pzi, sf, lf, pzf, kp, kp * rho_H, t_in, t_out
        )
        AGT_result = TT_wide_Texact(
            si, li, pzi, sf, lf, pzf, kp, kp * rho_H, t_in, t_out
        )
        print(numerical_result)
        if not torch.allclose(numerical_result, AGT_result, rtol=np.inf, atol=1e-04):
            diff = torch.abs(numerical_result - AGT_result)
            max_diff = torch.max(diff)
            self.fail(
                f"Max difference: {max_diff}\nNumerical result: {numerical_result}\nAGT result: {AGT_result}\nDifference: {diff}"
            )


if __name__ == "__main__":
    unittest.main()
