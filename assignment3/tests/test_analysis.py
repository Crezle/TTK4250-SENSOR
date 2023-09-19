# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import analysis as analysis
from solution.solu_usage_checker import UsageChecker


class Test_get_nis:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['analysis.get_nis']:
            values = tuple(kwargs.values())
            meas_pred, meas = values
            meas_pred_s, meas_s = deepcopy(values)

            ret = analysis.get_nis(meas_pred, meas)

            compare(meas_pred, meas_pred_s)
            compare(meas, meas_s)

            nis = ret
            nis_s = ret_s

            compare(nis, nis_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'analysis.get_nis'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            analysis.get_nis(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_get_nees:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['analysis.get_nees']:
            values = tuple(kwargs.values())
            state_est, x_gt = values
            state_est_s, x_gt_s = deepcopy(values)

            ret = analysis.get_nees(state_est, x_gt)

            compare(state_est, state_est_s)
            compare(x_gt, x_gt_s)

            NEES = ret
            NEES_s = ret_s

            compare(NEES, NEES_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'analysis.get_nees'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            analysis.get_nees(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
