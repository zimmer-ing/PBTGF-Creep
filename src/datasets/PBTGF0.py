# Copyright (c) 2025 Bernd Zimmering
# This file is part of the PBTGF-Creep project (https://github.com/zimmer-ing/PBTGF-Creep).
# Licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# If you use this code or the accompanying dataset, please cite:
# Klatt, E.; Zimmering, B.; Niggemann, O.; Rauter, N.:
# "Machine-Learning-Enabled Comparative Modelling of the Creep Behaviour of Unreinforced PBT and Short-Fibre Reinforced PBT Using Prony and Fractional Derivative Models", Appl. Mech. 2025.

from .dataset_base import DatasetBase
from pathlib import Path
import Constants as CONST

class PBTGF0(DatasetBase):
    def __init__(self):
        """
        Specific dataset class for PBTGF0.

        Automatically sets the folder path, column names,
        and cross-sectional area for this dataset.
        """
        super().__init__(
            folder_path=Path(CONST.DATA_PATH,"Tensile_Test_PBTGF0"),  # Path to the dataset folder
            name_strain="Strain_l_75_smooth",         # Column name for strain
            name_time="time",                         # Column name for time
            name_force="Force"                        # Column name for force
        )
        self.area = 42.0257  # Cross-sectional area in mm²

    def get_area(self):
        """
        Get the cross-sectional area of the samples.

        Returns:
            float: Cross-sectional area in mm².
        """
        return self.area
