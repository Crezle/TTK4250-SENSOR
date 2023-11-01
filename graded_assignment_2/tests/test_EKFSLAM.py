# pylint: skip-file
from copy import deepcopy
import sys
from pathlib import Path

project_dir = Path(__file__).parents[1]  # nopep8
sys.path.insert(0, str(project_dir.joinpath('src')))  # nopep8

from compare import compare
import EKFSLAM as EKFSLAM
import solution.EKFSLAM as solu_EKFSLAM
from solution.solu_usage_checker import UsageChecker


class Test_EKFSLAM__f:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['EKFSLAM.EKFSLAM.f']:
            values = tuple(kwargs.values())
            _self, x, u = values
            _self_s, x_s, u_s = deepcopy(values)

            ret = EKFSLAM.EKFSLAM.f(_self, x, u)
            ret_s = solu_EKFSLAM.EKFSLAM.f(_self_s, x_s, u_s)

            compare(_self, _self_s)
            compare(x, x_s)
            compare(u, u_s)

            xpred = ret
            xpred_s = ret_s

            compare(xpred, xpred_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'EKFSLAM.EKFSLAM.f'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            EKFSLAM.EKFSLAM.f(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_EKFSLAM__Fx:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['EKFSLAM.EKFSLAM.Fx']:
            values = tuple(kwargs.values())
            _self, x, u = values
            _self_s, x_s, u_s = deepcopy(values)

            ret = EKFSLAM.EKFSLAM.Fx(_self, x, u)
            ret_s = solu_EKFSLAM.EKFSLAM.Fx(_self_s, x_s, u_s)

            compare(_self, _self_s)
            compare(x, x_s)
            compare(u, u_s)

            Fx = ret
            Fx_s = ret_s

            compare(Fx, Fx_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'EKFSLAM.EKFSLAM.Fx'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            EKFSLAM.EKFSLAM.Fx(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_EKFSLAM__Fu:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['EKFSLAM.EKFSLAM.Fu']:
            values = tuple(kwargs.values())
            _self, x, u = values
            _self_s, x_s, u_s = deepcopy(values)

            ret = EKFSLAM.EKFSLAM.Fu(_self, x, u)
            ret_s = solu_EKFSLAM.EKFSLAM.Fu(_self_s, x_s, u_s)

            compare(_self, _self_s)
            compare(x, x_s)
            compare(u, u_s)

            Fu = ret
            Fu_s = ret_s

            compare(Fu, Fu_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'EKFSLAM.EKFSLAM.Fu'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            EKFSLAM.EKFSLAM.Fu(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_EKFSLAM__predict:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['EKFSLAM.EKFSLAM.predict']:
            values = tuple(kwargs.values())
            _self, eta, P, z_odo = values
            _self_s, eta_s, P_s, z_odo_s = deepcopy(values)

            ret = EKFSLAM.EKFSLAM.predict(_self, eta, P, z_odo)
            ret_s = solu_EKFSLAM.EKFSLAM.predict(_self_s, eta_s, P_s, z_odo_s)

            compare(_self, _self_s)
            compare(eta, eta_s)
            compare(P, P_s)
            compare(z_odo, z_odo_s)

            etapred, P = ret
            etapred_s, P_s = ret_s

            compare(etapred, etapred_s)
            compare(P, P_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'EKFSLAM.EKFSLAM.predict'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            EKFSLAM.EKFSLAM.predict(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_EKFSLAM__h:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['EKFSLAM.EKFSLAM.h']:
            values = tuple(kwargs.values())
            _self, eta = values
            _self_s, eta_s = deepcopy(values)

            ret = EKFSLAM.EKFSLAM.h(_self, eta)
            ret_s = solu_EKFSLAM.EKFSLAM.h(_self_s, eta_s)

            compare(_self, _self_s)
            compare(eta, eta_s)

            zpred = ret
            zpred_s = ret_s

            compare(zpred, zpred_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'EKFSLAM.EKFSLAM.h'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            EKFSLAM.EKFSLAM.h(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_EKFSLAM__h_jac:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['EKFSLAM.EKFSLAM.h_jac']:
            values = tuple(kwargs.values())
            _self, eta = values
            _self_s, eta_s = deepcopy(values)

            ret = EKFSLAM.EKFSLAM.h_jac(_self, eta)
            ret_s = solu_EKFSLAM.EKFSLAM.h_jac(_self_s, eta_s)

            compare(_self, _self_s)
            compare(eta, eta_s)

            H = ret
            H_s = ret_s

            compare(H, H_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'EKFSLAM.EKFSLAM.h_jac'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            EKFSLAM.EKFSLAM.h_jac(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_EKFSLAM__add_landmarks:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['EKFSLAM.EKFSLAM.add_landmarks']:
            values = tuple(kwargs.values())
            _self, eta, P, z = values
            _self_s, eta_s, P_s, z_s = deepcopy(values)

            ret = EKFSLAM.EKFSLAM.add_landmarks(_self, eta, P, z)
            ret_s = solu_EKFSLAM.EKFSLAM.add_landmarks(_self_s, eta_s, P_s, z_s)

            compare(_self, _self_s)
            compare(eta, eta_s)
            compare(P, P_s)
            compare(z, z_s)

            etaadded, Padded = ret
            etaadded_s, Padded_s = ret_s

            compare(etaadded, etaadded_s)
            compare(Padded, Padded_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'EKFSLAM.EKFSLAM.add_landmarks'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            EKFSLAM.EKFSLAM.add_landmarks(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


class Test_EKFSLAM__update:
    """Test class"""

    def test_output(self, test_data):
        """Tests if the function is correct by comparing the output
        with the output of the solution

        As python always use pass by reference, not by copy, it also checks if the
        input is changed (or not) in the same way as the in solution
        """
        for kwargs, ret_s in test_data['EKFSLAM.EKFSLAM.update']:
            values = tuple(kwargs.values())
            _self, eta, P, z = values
            _self_s, eta_s, P_s, z_s = deepcopy(values)

            ret = EKFSLAM.EKFSLAM.update(_self, eta, P, z)
            ret_s = solu_EKFSLAM.EKFSLAM.update(_self_s, eta_s, P_s, z_s)

            compare(_self, _self_s)
            compare(eta, eta_s)
            compare(P, P_s)
            compare(z, z_s)

            etaupd, Pupd, NIS, a = ret
            etaupd_s, Pupd_s, NIS_s, a_s = ret_s

            compare(etaupd, etaupd_s)
            compare(Pupd, Pupd_s)
            compare(NIS, NIS_s)
            compare(a, a_s)

    def test_solution_usage(self, test_data):
        """Tests if the solution is used in the function"""
        func_id = 'EKFSLAM.EKFSLAM.update'
        for kwargs, ret_s in test_data[func_id]:
            UsageChecker.reset_usage(func_id, None)
            EKFSLAM.EKFSLAM.update(**kwargs)
            msg = "The function uses the solution"
            assert not UsageChecker.is_used(func_id), msg


if __name__ == "__main__":
    import os
    import pytest
    os.environ["_PYTEST_RAISE"] = "1"
    pytest.main()
