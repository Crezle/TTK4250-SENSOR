# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import gaussian as gaussian
from solution.solu_usage_checker import UsageChecker


class Test_MultiVarGauss2d__get_transformed:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['gaussian.MultiVarGauss2d.get_transformed']:
            values = tuple(kwargs.values())
            _self, lin_transform = values
            _self_s, lin_transform_s = deepcopy(values)

            ret = gaussian.MultiVarGauss2d.get_transformed(
                _self, lin_transform)

            compare(_self, _self_s)
            compare(lin_transform, lin_transform_s)

            transformed = ret
            transformed_s = ret_s

            compare(transformed, transformed_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'gaussian.MultiVarGauss2d.get_transformed'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            gaussian.MultiVarGauss2d.get_transformed(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
