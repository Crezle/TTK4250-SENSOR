# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import filter as filter
from solution.solu_usage_checker import UsageChecker


class Test_FilterPDA__gate_zs:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['filter.FilterPDA.gate_zs']:
            values = tuple(kwargs.values())
            _self, z_est_pred, zs = values
            _self_s, z_est_pred_s, zs_s = deepcopy(values)

            ret = filter.FilterPDA.gate_zs(_self, z_est_pred, zs)

            compare(_self, _self_s)
            compare(z_est_pred, z_est_pred_s)
            compare(zs, zs_s)

            gated_indices, zs_gated = ret
            gated_indices_s, zs_gated_s = ret_s

            compare(gated_indices, gated_indices_s)
            compare(zs_gated, zs_gated_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'filter.FilterPDA.gate_zs'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            filter.FilterPDA.gate_zs(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_FilterPDA__get_assoc_probs:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['filter.FilterPDA.get_assoc_probs']:
            values = tuple(kwargs.values())
            _self, z_est_pred, zs = values
            _self_s, z_est_pred_s, zs_s = deepcopy(values)

            ret = filter.FilterPDA.get_assoc_probs(_self, z_est_pred, zs)

            compare(_self, _self_s)
            compare(z_est_pred, z_est_pred_s)
            compare(zs, zs_s)

            assoc_probs = ret
            assoc_probs_s = ret_s

            compare(assoc_probs, assoc_probs_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'filter.FilterPDA.get_assoc_probs'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            filter.FilterPDA.get_assoc_probs(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_FilterPDA__get_estimates:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['filter.FilterPDA.get_estimates']:
            values = tuple(kwargs.values())
            _self, x_est_pred, z_est_pred, zs_gated = values
            _self_s, x_est_pred_s, z_est_pred_s, zs_gated_s = deepcopy(values)

            ret = filter.FilterPDA.get_estimates(
                _self, x_est_pred, z_est_pred, zs_gated)

            compare(_self, _self_s)
            compare(x_est_pred, x_est_pred_s)
            compare(z_est_pred, z_est_pred_s)
            compare(zs_gated, zs_gated_s)

            x_ests = ret
            x_ests_s = ret_s

            compare(x_ests, x_ests_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'filter.FilterPDA.get_estimates'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            filter.FilterPDA.get_estimates(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_FilterPDA__step:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['filter.FilterPDA.step']:
            values = tuple(kwargs.values())
            _self, x_est_prev, zs, dt = values
            _self_s, x_est_prev_s, zs_s, dt_s = deepcopy(values)

            ret = filter.FilterPDA.step(_self, x_est_prev, zs, dt)

            compare(_self, _self_s)
            compare(x_est_prev, x_est_prev_s)
            compare(zs, zs_s)
            compare(dt, dt_s)

            x_est_upd, x_est_pred, z_est_pred, gated_indices = ret
            x_est_upd_s, x_est_pred_s, z_est_pred_s, gated_indices_s = ret_s

            compare(x_est_upd, x_est_upd_s)
            compare(x_est_pred, x_est_pred_s)
            compare(z_est_pred, z_est_pred_s)
            compare(gated_indices, gated_indices_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'filter.FilterPDA.step'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            filter.FilterPDA.step(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
