# Copyright (c) 2025 Bernd Zimmering
# This file is part of the PBTGF-Creep project (https://github.com/zimmer-ing/PBTGF-Creep).
# Licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# If you use this code or the accompanying dataset, please cite:
# Klatt, E.; Zimmering, B.; Niggemann, O.; Rauter, N.:
# "Machine-Learning-Enabled Comparative Modelling of the Creep Behaviour of Unreinforced PBT and Short-Fibre Reinforced PBT Using Prony and Fractional Derivative Models", Appl. Mech. 2025.
from abc import abstractmethod

import pandas as pd
import numpy as np
from pathlib import Path
import re

class DatasetBase:
    def __init__(self, folder_path, name_strain="Strain", name_time="Time", name_force="Force"):
        """
        Base class for datasets.

        Args:
            folder_path (str or Path): Path to the folder containing the dataset.
            name_strain (str): Column name for strain.
            name_time (str): Column name for time.
            name_force (str): Column name for force.
        """
        self.folder_path = Path(folder_path)
        self.name_strain = name_strain
        self.name_time = name_time
        self.name_force = name_force
        self.dataframes = None
        self.strain_tensor = None
        self.time_tensor = None
        self.force_tensor = None

    @staticmethod
    def _extract_numeric(file_name):
        """
        Extract numeric part from a file name for sorting.

        Args:
            file_name (str): File name to extract the number from.

        Returns:
            int: Extracted number or a high default value if no number is found.
        """
        match = re.search(r'\d+', file_name)
        return int(match.group()) if match else float('inf')

    def load_data(self):
        """
        Load data from the specified folder path and sort by numeric order.
        """
        # Sort CSV files numerically based on the extracted number
        csv_files = sorted(self.folder_path.glob('Sample_*.csv'), key=lambda x: self._extract_numeric(x.stem))
        self.dataframes = {csv_file.stem: pd.read_csv(csv_file) for csv_file in csv_files}
        print(f"Loaded {len(self.dataframes)} samples from {self.folder_path}")

    def prepare_data(self):
        """
        Prepare data by converting to tensors and applying necessary transformations.
        """
        if self.dataframes is None:
            raise ValueError("Data not loaded. Call `load_data` first.")

        # Extract tensors
        self.strain_tensor = np.array([df[self.name_strain].values for df in self.dataframes.values()])
        self.time_tensor = np.array([df[self.name_time].values for df in self.dataframes.values()])
        self.force_tensor = np.array([df[self.name_force].values for df in self.dataframes.values()])

        # Replace force with mean (constant for each sample)
        self.force_tensor = np.ones_like(self.force_tensor) * np.mean(self.force_tensor, axis=1, keepdims=True)

    def get_data(self):
        """
        Get the prepared tensors for strain, time, and force.

        Returns:
            tuple: (strain_tensor, time_tensor, force_tensor)
        """
        if self.strain_tensor is None or self.time_tensor is None or self.force_tensor is None:
            raise ValueError("Data not prepared. Call `prepare_data` first.")

        return self.strain_tensor, self.time_tensor, self.force_tensor

    @abstractmethod
    def get_area(self):
        """
        Get the cross-sectional area of the sample.

        Returns:
            float: Cross-sectional area in mm².
        """
        pass