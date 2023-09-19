# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import measurementmodels as measurementmodels
from solution.solu_usage_checker import UsageChecker


class Test_CartesianPosition2D__h:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['measurementmodels.CartesianPosition2D.h']:
            values = tuple(kwargs.values())
            _self, x = values
            _self_s, x_s = deepcopy(values)

            ret = measurementmodels.CartesianPosition2D.h(_self, x)

            compare(_self, _self_s)
            compare(x, x_s)

            x_h = ret
            x_h_s = ret_s

            compare(x_h, x_h_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'measurementmodels.CartesianPosition2D.h'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            measurementmodels.CartesianPosition2D.h(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_CartesianPosition2D__H:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['measurementmodels.CartesianPosition2D.H']:
            values = tuple(kwargs.values())
            _self, x = values
            _self_s, x_s = deepcopy(values)

            ret = measurementmodels.CartesianPosition2D.H(_self, x)

            compare(_self, _self_s)
            compare(x, x_s)

            H = ret
            H_s = ret_s

            compare(H, H_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'measurementmodels.CartesianPosition2D.H'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            measurementmodels.CartesianPosition2D.H(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_CartesianPosition2D__R:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['measurementmodels.CartesianPosition2D.R']:
            values = tuple(kwargs.values())
            _self, x = values
            _self_s, x_s = deepcopy(values)

            ret = measurementmodels.CartesianPosition2D.R(_self, x)

            compare(_self, _self_s)
            compare(x, x_s)

            R = ret
            R_s = ret_s

            compare(R, R_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'measurementmodels.CartesianPosition2D.R'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            measurementmodels.CartesianPosition2D.R(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_CartesianPosition2D__predict_measurement:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['measurementmodels.CartesianPosition2D.predict_measurement']:
            values = tuple(kwargs.values())
            _self, state_est = values
            _self_s, state_est_s = deepcopy(values)

            ret = measurementmodels.CartesianPosition2D.predict_measurement(
                _self, state_est)

            compare(_self, _self_s)
            compare(state_est, state_est_s)

            measure_pred_gauss = ret
            measure_pred_gauss_s = ret_s

            compare(measure_pred_gauss, measure_pred_gauss_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'measurementmodels.CartesianPosition2D.predict_measurement'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            measurementmodels.CartesianPosition2D.predict_measurement(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
