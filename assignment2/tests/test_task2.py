# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import task2 as task2
from solution.solu_usage_checker import UsageChecker


class Test_get_conds:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['task2.get_conds']:
            values = tuple(kwargs.values())
            state, sens_model_c, meas_c, sens_model_r, meas_r = values
            state_s, sens_model_c_s, meas_c_s, sens_model_r_s, meas_r_s = deepcopy(
                values)

            ret = task2.get_conds(state, sens_model_c,
                                  meas_c, sens_model_r, meas_r)

            compare(state, state_s)
            compare(sens_model_c, sens_model_c_s)
            compare(meas_c, meas_c_s)
            compare(sens_model_r, sens_model_r_s)
            compare(meas_r, meas_r_s)

            cond_c, cond_r = ret
            cond_c_s, cond_r_s = ret_s

            compare(cond_c, cond_c_s)
            compare(cond_r, cond_r_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'task2.get_conds'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            task2.get_conds(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_get_double_conds:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['task2.get_double_conds']:
            values = tuple(kwargs.values())
            state, sens_model_c, meas_c, sens_model_r, meas_r = values
            state_s, sens_model_c_s, meas_c_s, sens_model_r_s, meas_r_s = deepcopy(
                values)

            ret = task2.get_double_conds(
                state, sens_model_c, meas_c, sens_model_r, meas_r)

            compare(state, state_s)
            compare(sens_model_c, sens_model_c_s)
            compare(meas_c, meas_c_s)
            compare(sens_model_r, sens_model_r_s)
            compare(meas_r, meas_r_s)

            cond_cr, cond_rc = ret
            cond_cr_s, cond_rc_s = ret_s

            compare(cond_cr, cond_cr_s)
            compare(cond_rc, cond_rc_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'task2.get_double_conds'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            task2.get_double_conds(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_get_prob_over_line:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['task2.get_prob_over_line']:
            values = tuple(kwargs.values())
            gauss, = values
            gauss_s, = deepcopy(values)

            ret = task2.get_prob_over_line(gauss,)

            compare(gauss, gauss_s)

            prob = ret
            prob_s = ret_s

            compare(prob, prob_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'task2.get_prob_over_line'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            task2.get_prob_over_line(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
