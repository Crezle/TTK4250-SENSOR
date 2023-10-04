from dataclasses import dataclass
from typing import Sequence
from senfuslib import MultiVarGauss, DynamicModel, SensorModel, TimeSequence
import numpy as np
import tqdm
import logging

from states import StateCV, MeasPos
from models import ModelImm
from sensors import SensorPos
from gaussian_mixture import GaussianMixture
from solution import filter as filter_solu


@dataclass
class OutputEKF:
    x_est_upd: MultiVarGauss[StateCV]
    x_est_pred: MultiVarGauss[StateCV]
    z_est_pred: MultiVarGauss[MeasPos]


@dataclass
class EKF:
    dynamic_model: DynamicModel[StateCV]
    sensor_model: SensorModel[MeasPos]

    def step(self,
             x_est_prev: MultiVarGauss[StateCV],
             z: MeasPos,
             dt: float) -> OutputEKF:
        """Perform one EKF update step."""
        x_est_pred = self.dynamic_model.pred_from_est(x_est_prev, dt)
        z_est_pred = self.sensor_model.pred_from_est(x_est_pred)

        H_mat = self.sensor_model.H(x_est_pred.mean)
        P_mat = x_est_pred.cov
        S_mat = z_est_pred.cov

        kalman_gain = P_mat @ H_mat.T @ np.linalg.inv(S_mat)
        innovation = z - z_est_pred.mean

        state_upd_mean = x_est_pred.mean + kalman_gain @ innovation
        state_upd_cov = P_mat - kalman_gain @ H_mat @ P_mat

        x_est_upd = MultiVarGauss(state_upd_mean, state_upd_cov)

        return OutputEKF(x_est_upd, x_est_pred, z_est_pred)


@dataclass
class FilterIMM:
    dynamic_model: ModelImm
    sensor_model: SensorPos

    def calculate_mixings(self, x_est_prev: GaussianMixture[StateCV],
                          dt: float) -> np.ndarray:
        """Calculate the mixing probabilities, following step 1 in (6.4.1).

        The output should be on the following from:
        $mixing_probs[s_{k-1}, s_k] = \mu_{s_{k-1}|s_k}$
        """
        pi_mat = self.dynamic_model.get_pi_mat_d(dt)  # the pi in (6.6)
        prev_weights = x_est_prev.weights  # \mathbf{p}_{k-1}

        # print(f'PI_MAT SHAPE: {pi_mat.shape}')
        # print(f'PREV_WEIGHT SHAPE: {prev_weights.shape}')
        
        mixing_probs = pi_mat * prev_weights.reshape(-1, 1)

        for col in range(len(mixing_probs)):
            mixing_probs[:, col] = mixing_probs[:, col] / np.sum(mixing_probs[:, col])

        
        
        
        return mixing_probs

    def mixing(self,
               x_est_prev: GaussianMixture[StateCV],
               mixing_probs: GaussianMixture[StateCV] #NOTE: Why does this state that it is a GaussianMixture, but it is a matrix?
               ) -> Sequence[MultiVarGauss[StateCV]]:
        """Calculate the moment-based approximations, 
        following step 2 in (6.4.1). 
        Should return a gaussian with mean=(6.34) and cov=(6.35).

        Hint: Create a GaussianMixture for each mode density (6.33), 
        and use .reduce() to calculate (6.34) and (6.35).
        """
        moment_based_preds = []
        for i in range(len(x_est_prev)):
            mixture = GaussianMixture[StateCV](
                weights=mixing_probs[:, i],
                gaussians=x_est_prev.gaussians) # TODO (GaussianMixture)
            momend_based_pred = mixture.reduce()  # TODO (MultiVarGauss)
            moment_based_preds.append(momend_based_pred)

        return moment_based_preds

    def mode_match_filter(self,
                          moment_based_preds: GaussianMixture[StateCV],
                          z: MeasPos,
                          dt: float
                          ) -> Sequence[OutputEKF]:
        """Calculate the mode-match filter outputs (6.36),
        following step 3 in (6.4.1). 

        Hint: Use the EKF class from the top of this file.
        The last part (6.37) is not part of this
        method and is done later."""
        ekf_outs = []

        for i, x_prev in enumerate(moment_based_preds):
            
            ekf = EKF(dynamic_model=self.dynamic_model.models[i], sensor_model=self.sensor_model)
            out_ekf = ekf.step(x_prev, z, dt)  # TODO (OutputEKF)
            ekf_outs.append(out_ekf)

        return ekf_outs

    def update_probabilities(self,
                             ekf_outs: Sequence[OutputEKF],
                             z: MeasPos,
                             dt: float,
                             weights: np.ndarray
                             ) -> np.ndarray:
        """Update the mixing probabilities,
        using (6.37) from step 3 and (6.38) from step 4 in (6.4.1).

        Hint: Use (6.6)
        """
        pi_mat = self.dynamic_model.get_pi_mat_d(dt)

        weights_pred = pi_mat.T @ weights.reshape(-1, 1)  # TODO, use (6.6)

        z_probs = np.array([ekf_outs[i].z_est_pred.pdf(z) for i in range(len(ekf_outs))])  # TODO

        
        weights_upd = z_probs * weights_pred.flatten() / np.sum(z_probs * weights_pred.flatten())  # TODO
        
        return weights_upd

    def step(self,
             x_est_prev: GaussianMixture[StateCV],
             z: MeasPos,
             dt) -> GaussianMixture[StateCV]:
        """Perform one step of the IMM filter."""
        mixing_probs = self.calculate_mixings(x_est_prev, dt)  # TODO
        moment_based_preds = self.mixing(x_est_prev, mixing_probs)  # TODO
        ekf_outs = self.mode_match_filter(moment_based_preds, z, dt)  # TODO
        weights_upd = self.update_probabilities(ekf_outs, z, dt, x_est_prev.weights)  # TODO
        
        x_est_upd = GaussianMixture(weights_upd,
                                    [out.x_est_upd for out in ekf_outs])  # TODO
        x_est_pred = GaussianMixture(x_est_prev.weights,
                                        [out.x_est_pred for out in ekf_outs])
        z_est_pred = GaussianMixture(x_est_prev.weights,
                                        [out.z_est_pred for out in ekf_outs])


        return x_est_upd, x_est_pred, z_est_pred

    def run(self, x0_est: GaussianMixture[StateCV], zs: TimeSequence[MeasPos]
            ) -> TimeSequence[GaussianMixture[StateCV]]:
        """Run the IMM filter."""
        logging.info("Running IMM filter")
        x_est_upds = TimeSequence()
        x_est_preds = TimeSequence()
        z_est_preds = TimeSequence()
        x_est_upds.insert(0, x0_est)
        t_prev = 0
        for t, z in tqdm.tqdm(zs.items(), total=len(zs)):
            t_prev, x_est_prev = x_est_upds[-1]
            dt = np.round(t-t_prev, 8)

            x_est_upd, x_est_pred, z_est_pred = self.step(x_est_prev, z, dt)
            x_est_upds.insert(t, x_est_upd)
            x_est_preds.insert(t, x_est_pred)
            z_est_preds.insert(t, z_est_pred)
        return x_est_upds, x_est_preds, z_est_preds
