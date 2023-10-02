# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import filter as filter
from solution.solu_usage_checker import UsageChecker


class Test_FilterIMM__calculate_mixings:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['filter.FilterIMM.calculate_mixings']:
            values = tuple(kwargs.values())
            _self, x_est_prev, dt = values
            _self_s, x_est_prev_s, dt_s = deepcopy(values)

            ret = filter.FilterIMM.calculate_mixings(_self, x_est_prev, dt)

            compare(_self, _self_s)
            compare(x_est_prev, x_est_prev_s)
            compare(dt, dt_s)

            mixing_probs = ret
            mixing_probs_s = ret_s

            compare(mixing_probs, mixing_probs_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'filter.FilterIMM.calculate_mixings'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            filter.FilterIMM.calculate_mixings(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_FilterIMM__mixing:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['filter.FilterIMM.mixing']:
            values = tuple(kwargs.values())
            _self, x_est_prev, mixing_probs = values
            _self_s, x_est_prev_s, mixing_probs_s = deepcopy(values)

            ret = filter.FilterIMM.mixing(_self, x_est_prev, mixing_probs)

            compare(_self, _self_s)
            compare(x_est_prev, x_est_prev_s)
            compare(mixing_probs, mixing_probs_s)

            moment_based_preds = ret
            moment_based_preds_s = ret_s

            compare(moment_based_preds, moment_based_preds_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'filter.FilterIMM.mixing'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            filter.FilterIMM.mixing(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_FilterIMM__mode_match_filter:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['filter.FilterIMM.mode_match_filter']:
            values = tuple(kwargs.values())
            _self, moment_based_preds, z, dt = values
            _self_s, moment_based_preds_s, z_s, dt_s = deepcopy(values)

            ret = filter.FilterIMM.mode_match_filter(
                _self, moment_based_preds, z, dt)

            compare(_self, _self_s)
            compare(moment_based_preds, moment_based_preds_s)
            compare(z, z_s)
            compare(dt, dt_s)

            ekf_outs = ret
            ekf_outs_s = ret_s

            compare(ekf_outs, ekf_outs_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'filter.FilterIMM.mode_match_filter'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            filter.FilterIMM.mode_match_filter(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_FilterIMM__update_probabilities:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['filter.FilterIMM.update_probabilities']:
            values = tuple(kwargs.values())
            _self, ekf_outs, z, dt, weights = values
            _self_s, ekf_outs_s, z_s, dt_s, weights_s = deepcopy(values)

            ret = filter.FilterIMM.update_probabilities(
                _self, ekf_outs, z, dt, weights)

            compare(_self, _self_s)
            compare(ekf_outs, ekf_outs_s)
            compare(z, z_s)
            compare(dt, dt_s)
            compare(weights, weights_s)

            weights_upd = ret
            weights_upd_s = ret_s

            compare(weights_upd, weights_upd_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'filter.FilterIMM.update_probabilities'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            filter.FilterIMM.update_probabilities(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_FilterIMM__step:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['filter.FilterIMM.step']:
            values = tuple(kwargs.values())
            _self, x_est_prev, z, dt = values
            _self_s, x_est_prev_s, z_s, dt_s = deepcopy(values)

            ret = filter.FilterIMM.step(_self, x_est_prev, z, dt)

            compare(_self, _self_s)
            compare(x_est_prev, x_est_prev_s)
            compare(z, z_s)
            compare(dt, dt_s)

            x_est_upd, x_est_pred, z_est_pred = ret
            x_est_upd_s, x_est_pred_s, z_est_pred_s = ret_s

            compare(x_est_upd, x_est_upd_s)
            compare(x_est_pred, x_est_pred_s)
            compare(z_est_pred, z_est_pred_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'filter.FilterIMM.step'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            filter.FilterIMM.step(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
