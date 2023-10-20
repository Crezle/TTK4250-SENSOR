# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import quaternion as quaternion
from solution.solu_usage_checker import UsageChecker


class Test_RotationQuaterion__multiply:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['quaternion.RotationQuaterion.multiply']:
            values = tuple(kwargs.values())
            _self, other = values
            _self_s, other_s = deepcopy(values)

            ret = quaternion.RotationQuaterion.multiply(_self, other)

            compare(_self, _self_s)
            compare(other, other_s)

            quaternion_product = ret
            quaternion_product_s = ret_s

            compare(quaternion_product, quaternion_product_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'quaternion.RotationQuaterion.multiply'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            quaternion.RotationQuaterion.multiply(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_RotationQuaterion__conjugate:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['quaternion.RotationQuaterion.conjugate']:
            values = tuple(kwargs.values())
            _self, = values
            _self_s, = deepcopy(values)

            ret = quaternion.RotationQuaterion.conjugate(_self,)

            compare(_self, _self_s)

            conj = ret
            conj_s = ret_s

            compare(conj, conj_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'quaternion.RotationQuaterion.conjugate'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            quaternion.RotationQuaterion.conjugate(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
