# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import initialize as initialize
from solution.solu_usage_checker import UsageChecker


class Test_get_init_CV_state:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['initialize.get_init_CV_state']:
            values = tuple(kwargs.values())
            meas0, meas1, ekf_params = values
            meas0_s, meas1_s, ekf_params_s = deepcopy(values)

            ret = initialize.get_init_CV_state(meas0, meas1, ekf_params)

            compare(meas0, meas0_s)
            compare(meas1, meas1_s)
            compare(ekf_params, ekf_params_s)

            init_state = ret
            init_state_s = ret_s

            compare(init_state, init_state_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'initialize.get_init_CV_state'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            initialize.get_init_CV_state(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
