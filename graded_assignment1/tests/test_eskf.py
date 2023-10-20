# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import eskf as eskf
from solution.solu_usage_checker import UsageChecker


class Test_ESKF__predict_from_imu:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['eskf.ESKF.predict_from_imu']:
            values = tuple(kwargs.values())
            _self, x_est_prev, z_imu, dt = values
            _self_s, x_est_prev_s, z_imu_s, dt_s = deepcopy(values)

            ret = eskf.ESKF.predict_from_imu(_self, x_est_prev, z_imu, dt)

            compare(_self, _self_s)
            compare(x_est_prev, x_est_prev_s)
            compare(z_imu, z_imu_s)
            compare(dt, dt_s)

            x_est_prev = ret
            x_est_prev_s = ret_s

            compare(x_est_prev, x_est_prev_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'eskf.ESKF.predict_from_imu'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            eskf.ESKF.predict_from_imu(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_ESKF__update_err_from_gnss:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['eskf.ESKF.update_err_from_gnss']:
            values = tuple(kwargs.values())
            _self, x_est_pred, z_est_pred, z_gnss = values
            _self_s, x_est_pred_s, z_est_pred_s, z_gnss_s = deepcopy(values)

            ret = eskf.ESKF.update_err_from_gnss(
                _self, x_est_pred, z_est_pred, z_gnss)

            compare(_self, _self_s)
            compare(x_est_pred, x_est_pred_s)
            compare(z_est_pred, z_est_pred_s)
            compare(z_gnss, z_gnss_s)

            x_est_upd_err = ret
            x_est_upd_err_s = ret_s

            compare(x_est_upd_err, x_est_upd_err_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'eskf.ESKF.update_err_from_gnss'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            eskf.ESKF.update_err_from_gnss(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_ESKF__inject:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['eskf.ESKF.inject']:
            values = tuple(kwargs.values())
            _self, x_est_nom, x_est_err = values
            _self_s, x_est_nom_s, x_est_err_s = deepcopy(values)

            ret = eskf.ESKF.inject(_self, x_est_nom, x_est_err)

            compare(_self, _self_s)
            compare(x_est_nom, x_est_nom_s)
            compare(x_est_err, x_est_err_s)

            x_est_inj = ret
            x_est_inj_s = ret_s

            compare(x_est_inj, x_est_inj_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'eskf.ESKF.inject'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            eskf.ESKF.inject(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_ESKF__update_from_gnss:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['eskf.ESKF.update_from_gnss']:
            values = tuple(kwargs.values())
            _self, x_est_pred, z_gnss = values
            _self_s, x_est_pred_s, z_gnss_s = deepcopy(values)

            ret = eskf.ESKF.update_from_gnss(_self, x_est_pred, z_gnss)

            compare(_self, _self_s)
            compare(x_est_pred, x_est_pred_s)
            compare(z_gnss, z_gnss_s)

            x_est_upd, z_est_pred = ret
            x_est_upd_s, z_est_pred_s = ret_s

            compare(x_est_upd, x_est_upd_s)
            compare(z_est_pred, z_est_pred_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'eskf.ESKF.update_from_gnss'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            eskf.ESKF.update_from_gnss(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
