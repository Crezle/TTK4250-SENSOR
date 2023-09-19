# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import ekf as ekf
from solution.solu_usage_checker import UsageChecker


class Test_ExtendedKalmanFilter__step:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['ekf.ExtendedKalmanFilter.step']:
            values = tuple(kwargs.values())
            _self, state_old, meas = values
            _self_s, state_old_s, meas_s = deepcopy(values)

            ret = ekf.ExtendedKalmanFilter.step(_self, state_old, meas)

            compare(_self, _self_s)
            compare(state_old, state_old_s)
            compare(meas, meas_s)

            state_upd, state_pred, meas_pred = ret
            state_upd_s, state_pred_s, meas_pred_s = ret_s

            compare(state_upd, state_upd_s)
            compare(state_pred, state_pred_s)
            compare(meas_pred, meas_pred_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'ekf.ExtendedKalmanFilter.step'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            ekf.ExtendedKalmanFilter.step(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
