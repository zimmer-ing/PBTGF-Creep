# Copyright (c) 2025 Bernd Zimmering
# This file is part of the PBTGF-Creep project (https://github.com/zimmer-ing/PBTGF-Creep).
# Licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# If you use this code or the accompanying dataset, please cite:
# Klatt, E.; Zimmering, B.; Niggemann, O.; Rauter, N.:
# "Machine-Learning-Enabled Comparative Modelling of the Creep Behaviour of Unreinforced PBT and Short-Fibre Reinforced PBT Using Prony and Fractional Derivative Models", Appl. Mech. 2025.
from pathlib import Path
import pandas as pd
from src.models.models_base import BaseModel
import random
from scipy.optimize import minimize
import numpy as np
seed = 42
random.seed(seed)
np.random.seed(seed)


class PronyModel(BaseModel):
    """
    A Prony model class for fitting multiple samples with a specified number of Prony terms.
    This model attempts to fit a Prony series to stiffness data derived from strain and force measurements.

    Parameters
    ----------
    n_terms : int
        Number of Prony terms.
    results_path : Path
        Path where the results will be saved.
    area : float
        Cross-sectional area of the sample in mm².
    """
    def __init__(self, n_terms, results_path, area):
        super().__init__(n_terms, results_path, area)
        self.area = area

    @staticmethod
    def prony_series(t, K_eq, *params):
        """
        Computes the Prony series for given time points.

        Prony series:
        stiffness(t) = K_eq + Σ [K_i * exp(-t / tau_i)] for i = 1 to n_terms

        Parameters
        ----------
        t : np.ndarray
            Time values.
        K_eq : float
            Equilibrium modulus (long-term stiffness).
        params : tuple
            A sequence of K_i and tau_i for each term (K_1, tau_1, K_2, tau_2, ...).

        Returns
        -------
        np.ndarray
            Computed stiffness values at each time point.
        """
        result = K_eq
        for i in range(0, len(params), 2):
            K_i = params[i]
            tau_i = params[i + 1]
            result += K_i * np.exp(-t / tau_i)
        return result

    def fit_sample(self, time, stiffness):
        """
        Fit the Prony series to the stiffness data of a single sample.

        This method sets up an optimization problem to minimize the sum of squared residuals
        between the Prony series and the given stiffness data. It also enforces some constraints:
        - K_i should be non-increasing with i (K_{i} >= K_{i+1})
        - tau_i should be non-increasing with i (tau_{i} >= tau_{i+1})

        Parameters
        ----------
        time : np.ndarray
            Time values for the sample.
        stiffness : np.ndarray
            Stiffness values for the sample at the corresponding time points.

        Returns
        -------
        dict
            Dictionary of fitted parameters including K_eq, K_i, and tau_i for each Prony term.
        """



        def weighting_function(time_array, alpha=3.0, tau=30.0, baseline=1.0):
            """
            Exponential weighting: w(t) = alpha * exp(-t / tau) + baseline

            Parameters:
            - time_array : np.ndarray
            - alpha : float      → initial extra weight at t=0
            - tau : float        → controls decay rate (like time constant)
            - baseline : float   → weight for large t

            Returns:
            - weights : np.ndarray
            """
            weights = alpha * np.exp(-time_array / tau) + baseline
            return weights
        # Define the loss function (sum of squared residuals)
        def loss_func(params):
            K_eq = params[0]
            prony_params = params[1:]
            residuals = self.prony_series(time, K_eq, *prony_params) - stiffness
            weights = weighting_function(time, alpha=10.0, tau=15, baseline=1.0)
            return np.sum((weights * residuals) ** 2)



        # Initial guess:
        # Start with K_eq as 0.8 * max(stiffness), and gradually smaller K_i and tau_i.
        diff_stiffness = np.max(stiffness) - np.min(stiffness)
        initial_guess = [np.min(stiffness)]

        # Generate weights: [n, n-1, ..., 1]
        weights = np.arange(self.n_terms, 0, -1)
        weights = weights / np.sum(weights)  # normalize to sum = 1

        for i in range(self.n_terms):
            K_i = diff_stiffness * weights[i]  # larger Ks at the beginning
            tau_i = 10*self.n_terms * (1 / (i + 1))  # or use another logic
            initial_guess += [K_i, tau_i]

        # Perform the optimization with constraints
        # Bounds ensure parameters stay within a reasonable range

        result = minimize(
            loss_func,
            initial_guess,
            bounds=[(0, np.max(stiffness) * 10)] + [(0, np.max(stiffness) * 10), (1, 1000)] * self.n_terms,
            #constraints={'type': 'ineq', 'fun': constraint},
            method='L-BFGS-B',
            options={'maxiter': 1000, 'disp': False}
        )
        if not result.success:
            print(f"Warning: Optimization failed – {result.message}")
        elif result.nit >= 1000:
            print("Warning: Max iterations reached")
        params = result.x

        # Extract the fitted parameters into a dictionary
        fitted = {'K_eq': params[0]}
        params_terms = params[1:]
        for i in range(self.n_terms):
            fitted[f"K_{i + 1}"] = params_terms[2 * i]
            fitted[f"tau_{i + 1}"] = params_terms[2 * i + 1]

        return fitted

    def fit(self, time_tensor, strain_tensor, force_tensor):
        """
        Fit the Prony series to multiple samples in the dataset.

        Steps:
        1. Compute the global stiffness from strain and force data.
        2. Fit each sample's stiffness-time data with the Prony model.

        Parameters
        ----------
        time_tensor : np.ndarray
            Time data for all samples, shape (n_samples, n_timesteps).
        strain_tensor : np.ndarray
            Strain data for all samples, shape (n_samples, n_timesteps).
        force_tensor : np.ndarray
            Force data for all samples, shape (n_samples, n_timesteps).

        Returns
        -------
        list
            List of parameter dictionaries for each sample.
        """
        # Convert strain-force data to stiffness
        E_global = self.calculate_global_stiffness(strain_tensor, force_tensor)

        all_params = []
        for sample_idx in range(time_tensor.shape[0]):
            time = time_tensor[sample_idx]
            stiffness = E_global[sample_idx]

            # Fit the sample using the previously defined method
            params = self.fit_sample(time, stiffness)
            all_params.append(params)

        return all_params

    def predict(self, time, force, fitted_params):
        """
        Predict strain based on the fitted Prony parameters.

        For each sample:
        1. Compute the stiffness over time using the Prony series.
        2. Convert force to strain: strain = (force / (stiffness * area)) * 100.

        Parameters
        ----------
        time : np.ndarray
            Time values for prediction, shape (n_samples, n_timesteps).
        force : np.ndarray
            Force values for prediction, shape (n_samples, n_timesteps).
        fitted_params : pd.DataFrame
            DataFrame containing the fitted parameters (K_eq, K_i, tau_i) for each sample.

        Returns
        -------
        np.ndarray
            Predicted strain values for all samples, shape (n_samples, n_timesteps).
        """
        predictions = []

        for sample_idx, params in enumerate(fitted_params.to_dict(orient="records")):
            # Extract the Prony parameters
            K_eq = params["K_eq"]
            prony_params = []
            for i in range(self.n_terms):
                prony_params.append(params[f"K_{i + 1}"])
                prony_params.append(params[f"tau_{i + 1}"])

            # Compute stiffness using the Prony series
            stiffness = self.prony_series(time[sample_idx], K_eq, *prony_params)

            # Average force for this sample
            force_mean = force[sample_idx].mean()

            # Calculate predicted strain (in %)
            predicted_strain = (force_mean / (stiffness * self.area)) * 100
            predictions.append(predicted_strain)

        return np.array(predictions)