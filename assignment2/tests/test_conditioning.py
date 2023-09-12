# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import conditioning as conditioning
from solution.solu_usage_checker import UsageChecker


class Test_get_cond_state:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['conditioning.get_cond_state']:
            values = tuple(kwargs.values())
            state, sens_modl, meas = values
            state_s, sens_modl_s, meas_s = deepcopy(values)

            ret = conditioning.get_cond_state(state, sens_modl, meas)

            compare(state, state_s)
            compare(sens_modl, sens_modl_s)
            compare(meas, meas_s)

            cond_state = ret
            cond_state_s = ret_s

            compare(cond_state, cond_state_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'conditioning.get_cond_state'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            conditioning.get_cond_state(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
