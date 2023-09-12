# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import sensor_model as sensor_model
from solution.solu_usage_checker import UsageChecker


class Test_LinearSensorModel2d__get_pred_meas:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['sensor_model.LinearSensorModel2d.get_pred_meas']:
            values = tuple(kwargs.values())
            _self, state_est = values
            _self_s, state_est_s = deepcopy(values)

            ret = sensor_model.LinearSensorModel2d.get_pred_meas(
                _self, state_est)

            compare(_self, _self_s)
            compare(state_est, state_est_s)

            pred_meas = ret
            pred_meas_s = ret_s

            compare(pred_meas, pred_meas_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'sensor_model.LinearSensorModel2d.get_pred_meas'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            sensor_model.LinearSensorModel2d.get_pred_meas(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
