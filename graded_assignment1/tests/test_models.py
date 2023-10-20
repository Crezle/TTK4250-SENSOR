# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import models as models
from solution.solu_usage_checker import UsageChecker


class Test_ModelIMU__correct_z_imu:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['models.ModelIMU.correct_z_imu']:
            values = tuple(kwargs.values())
            _self, x_est_nom, z_imu = values
            _self_s, x_est_nom_s, z_imu_s = deepcopy(values)

            ret = models.ModelIMU.correct_z_imu(_self, x_est_nom, z_imu)

            compare(_self, _self_s)
            compare(x_est_nom, x_est_nom_s)
            compare(z_imu, z_imu_s)

            z_corr = ret
            z_corr_s = ret_s

            compare(z_corr, z_corr_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'models.ModelIMU.correct_z_imu'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            models.ModelIMU.correct_z_imu(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_ModelIMU__predict_nom:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['models.ModelIMU.predict_nom']:
            values = tuple(kwargs.values())
            _self, x_est_nom, z_corr, dt = values
            _self_s, x_est_nom_s, z_corr_s, dt_s = deepcopy(values)

            ret = models.ModelIMU.predict_nom(_self, x_est_nom, z_corr, dt)

            compare(_self, _self_s)
            compare(x_est_nom, x_est_nom_s)
            compare(z_corr, z_corr_s)
            compare(dt, dt_s)

            x_nom_pred = ret
            x_nom_pred_s = ret_s

            compare(x_nom_pred, x_nom_pred_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'models.ModelIMU.predict_nom'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            models.ModelIMU.predict_nom(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_ModelIMU__A_c:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['models.ModelIMU.A_c']:
            values = tuple(kwargs.values())
            _self, x_est_nom, z_corr = values
            _self_s, x_est_nom_s, z_corr_s = deepcopy(values)

            ret = models.ModelIMU.A_c(_self, x_est_nom, z_corr)

            compare(_self, _self_s)
            compare(x_est_nom, x_est_nom_s)
            compare(z_corr, z_corr_s)

            A_c = ret
            A_c_s = ret_s

            compare(A_c, A_c_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'models.ModelIMU.A_c'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            models.ModelIMU.A_c(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_ModelIMU__get_error_G_c:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['models.ModelIMU.get_error_G_c']:
            values = tuple(kwargs.values())
            _self, x_est_nom = values
            _self_s, x_est_nom_s = deepcopy(values)

            ret = models.ModelIMU.get_error_G_c(_self, x_est_nom)

            compare(_self, _self_s)
            compare(x_est_nom, x_est_nom_s)

            G_c = ret
            G_c_s = ret_s

            compare(G_c, G_c_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'models.ModelIMU.get_error_G_c'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            models.ModelIMU.get_error_G_c(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_ModelIMU__get_discrete_error_diff:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['models.ModelIMU.get_discrete_error_diff']:
            values = tuple(kwargs.values())
            _self, x_est_nom, z_corr, dt = values
            _self_s, x_est_nom_s, z_corr_s, dt_s = deepcopy(values)

            ret = models.ModelIMU.get_discrete_error_diff(
                _self, x_est_nom, z_corr, dt)

            compare(_self, _self_s)
            compare(x_est_nom, x_est_nom_s)
            compare(z_corr, z_corr_s)
            compare(dt, dt_s)

            A_d, GQGT_d = ret
            A_d_s, GQGT_d_s = ret_s

            compare(A_d, A_d_s)
            compare(GQGT_d, GQGT_d_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'models.ModelIMU.get_discrete_error_diff'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            models.ModelIMU.get_discrete_error_diff(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_ModelIMU__predict_err:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['models.ModelIMU.predict_err']:
            values = tuple(kwargs.values())
            _self, x_est_prev, z_corr, dt = values
            _self_s, x_est_prev_s, z_corr_s, dt_s = deepcopy(values)

            ret = models.ModelIMU.predict_err(_self, x_est_prev, z_corr, dt)

            compare(_self, _self_s)
            compare(x_est_prev, x_est_prev_s)
            compare(z_corr, z_corr_s)
            compare(dt, dt_s)

            x_err_pred = ret
            x_err_pred_s = ret_s

            compare(x_err_pred, x_err_pred_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'models.ModelIMU.predict_err'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            models.ModelIMU.predict_err(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
