# Copyright (c) 2025 Bernd Zimmering
# This file is part of the PBTGF-Creep project (https://github.com/zimmer-ing/PBTGF-Creep).
# Licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# If you use this code or the accompanying dataset, please cite:
# Klatt, E.; Zimmering, B.; Niggemann, O.; Rauter, N.:
# "Machine-Learning-Enabled Comparative Modelling of the Creep Behaviour of Unreinforced PBT and Short-Fibre Reinforced PBT Using Prony and Fractional Derivative Models", Appl. Mech. 2025.
from abc import ABC, abstractmethod
from pathlib import Path
import numpy as np
import pandas as pd

class BaseModel(ABC):
    def __init__(self,  n_terms, results_path, area):
        """
        Base class for models.

        Args:
            n_terms (int): Number of terms in the model.
            results_path (str): Path to save results.
            area (float): Cross-sectional area in mm².
        """
        self.n_terms = n_terms
        self.results_path = Path(results_path)
        self.area = area
        self.params = None
        self.results_path.mkdir(parents=True, exist_ok=True)

    def calculate_stress(self, force_tensor):
        """
        Calculate stress using force and cross-sectional area.

        Args:
            force_tensor (np.ndarray): Force values.

        Returns:
            np.ndarray: Stress values.
        """
        return force_tensor / self.area  # MPa

    def calculate_global_stiffness(self, strain_tensor, force_tensor):
        """
        Calculate global stiffness.

        Args:
            strain_tensor (np.ndarray): Strain values as percentages.
            force_tensor (np.ndarray): Force values.

        Returns:
            np.ndarray: Global stiffness.
        """
        strain_fraction = strain_tensor / 100.0  # Convert strain to fraction
        sigma = self.calculate_stress(force_tensor)  # Stress in MPa
        return sigma / strain_fraction  # Stiffness (MPa)



    @abstractmethod
    def fit(self, time, stiffness):
        """
        Abstract method to fit the model to data.
        Must be implemented in derived classes.
        """
        pass

    @abstractmethod
    def predict(self, time):
        """
        Abstract method to predict based on the model.
        Must be implemented in derived classes.
        """
        pass

    def evaluate(self, measured, predicted):
        """
        Evaluate the model using MSE and MAE.
        """
        mse = np.mean((measured - predicted) ** 2)
        mae = np.mean(np.abs(measured - predicted))
        return mse, mae