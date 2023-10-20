# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import sensors as sensors
from solution.solu_usage_checker import UsageChecker


class Test_SensorGNSS__H:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['sensors.SensorGNSS.H']:
            values = tuple(kwargs.values())
            _self, x_nom = values
            _self_s, x_nom_s = deepcopy(values)

            ret = sensors.SensorGNSS.H(_self, x_nom)

            compare(_self, _self_s)
            compare(x_nom, x_nom_s)

            H = ret
            H_s = ret_s

            compare(H, H_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'sensors.SensorGNSS.H'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            sensors.SensorGNSS.H(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_SensorGNSS__pred_from_est:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['sensors.SensorGNSS.pred_from_est']:
            values = tuple(kwargs.values())
            _self, x_est = values
            _self_s, x_est_s = deepcopy(values)

            ret = sensors.SensorGNSS.pred_from_est(_self, x_est)

            compare(_self, _self_s)
            compare(x_est, x_est_s)

            z_gnss_pred_gauss = ret
            z_gnss_pred_gauss_s = ret_s

            compare(z_gnss_pred_gauss, z_gnss_pred_gauss_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'sensors.SensorGNSS.pred_from_est'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            sensors.SensorGNSS.pred_from_est(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
