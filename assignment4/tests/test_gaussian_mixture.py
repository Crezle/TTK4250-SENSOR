# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import gaussian_mixture as gaussian_mixture
from solution.solu_usage_checker import UsageChecker


class Test_GaussianMixture__mean:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['gaussian_mixture.GaussianMixture.mean']:
            values = tuple(kwargs.values())
            _self, = values
            _self_s, = deepcopy(values)

            ret = gaussian_mixture.GaussianMixture.mean(_self,)

            compare(_self, _self_s)

            mean = ret
            mean_s = ret_s

            compare(mean, mean_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'gaussian_mixture.GaussianMixture.mean'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            gaussian_mixture.GaussianMixture.mean(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_GaussianMixture__cov:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['gaussian_mixture.GaussianMixture.cov']:
            values = tuple(kwargs.values())
            _self, = values
            _self_s, = deepcopy(values)

            ret = gaussian_mixture.GaussianMixture.cov(_self,)

            compare(_self, _self_s)

            cov = ret
            cov_s = ret_s

            compare(cov, cov_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'gaussian_mixture.GaussianMixture.cov'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            gaussian_mixture.GaussianMixture.cov(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
