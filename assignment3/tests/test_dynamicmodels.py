# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import dynamicmodels as dynamicmodels
from solution.solu_usage_checker import UsageChecker


class Test_WhitenoiseAcceleration2D__f:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['dynamicmodels.WhitenoiseAcceleration2D.f']:
            values = tuple(kwargs.values())
            _self, x, dt = values
            _self_s, x_s, dt_s = deepcopy(values)

            ret = dynamicmodels.WhitenoiseAcceleration2D.f(_self, x, dt)

            compare(_self, _self_s)
            compare(x, x_s)
            compare(dt, dt_s)

            x_next = ret
            x_next_s = ret_s

            compare(x_next, x_next_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'dynamicmodels.WhitenoiseAcceleration2D.f'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            dynamicmodels.WhitenoiseAcceleration2D.f(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_WhitenoiseAcceleration2D__F:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['dynamicmodels.WhitenoiseAcceleration2D.F']:
            values = tuple(kwargs.values())
            _self, x, dt = values
            _self_s, x_s, dt_s = deepcopy(values)

            ret = dynamicmodels.WhitenoiseAcceleration2D.F(_self, x, dt)

            compare(_self, _self_s)
            compare(x, x_s)
            compare(dt, dt_s)

            F = ret
            F_s = ret_s

            compare(F, F_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'dynamicmodels.WhitenoiseAcceleration2D.F'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            dynamicmodels.WhitenoiseAcceleration2D.F(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_WhitenoiseAcceleration2D__Q:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['dynamicmodels.WhitenoiseAcceleration2D.Q']:
            values = tuple(kwargs.values())
            _self, x, dt = values
            _self_s, x_s, dt_s = deepcopy(values)

            ret = dynamicmodels.WhitenoiseAcceleration2D.Q(_self, x, dt)

            compare(_self, _self_s)
            compare(x, x_s)
            compare(dt, dt_s)

            Q = ret
            Q_s = ret_s

            compare(Q, Q_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'dynamicmodels.WhitenoiseAcceleration2D.Q'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            dynamicmodels.WhitenoiseAcceleration2D.Q(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_WhitenoiseAcceleration2D__predict_state:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['dynamicmodels.WhitenoiseAcceleration2D.predict_state']:
            values = tuple(kwargs.values())
            _self, state_est, dt = values
            _self_s, state_est_s, dt_s = deepcopy(values)

            ret = dynamicmodels.WhitenoiseAcceleration2D.predict_state(
                _self, state_est, dt)

            compare(_self, _self_s)
            compare(state_est, state_est_s)
            compare(dt, dt_s)

            state_pred_gauss = ret
            state_pred_gauss_s = ret_s

            compare(state_pred_gauss, state_pred_gauss_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'dynamicmodels.WhitenoiseAcceleration2D.predict_state'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            dynamicmodels.WhitenoiseAcceleration2D.predict_state(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
