# Copyright (c) 2025 Bernd Zimmering
# This file is part of the PBTGF-Creep project (https://github.com/zimmer-ing/PBTGF-Creep).
# Licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# If you use this code or the accompanying dataset, please cite:
# Klatt, E.; Zimmering, B.; Niggemann, O.; Rauter, N.:
# "Machine-Learning-Enabled Comparative Modelling of the Creep Behaviour of Unreinforced PBT and Short-Fibre Reinforced PBT Using Prony and Fractional Derivative Models", Appl. Mech. 2025.
import h5py
import pandas as pd
from pathlib import Path
import Constants as CONST


# Create directories or clear existing ones
def prepare_output_directory(output_path):
    """
    Creates the output directory or clears it if it already exists.
    """
    if output_path.exists():
        for file in output_path.glob('*.csv'):
            file.unlink()
    else:
        output_path.mkdir(parents=True)


# Function to load data from a .mat file
def load_dataset(file_path):
    """
    Loads the data from a .mat file into a dictionary.

    Args:
        file_path (Path): Path to the .mat file.

    Returns:
        dict: Dictionary containing the dataset.
    """
    data = {}
    with h5py.File(file_path, 'r') as mat_data:
        for key in mat_data.keys():
            if isinstance(mat_data[key], h5py.Dataset):
                data[key] = mat_data[key][()]
            elif isinstance(mat_data[key], h5py.Group):
                for sub_key in mat_data[key].keys():
                    if isinstance(mat_data[key][sub_key], h5py.Dataset):
                        data[sub_key] = mat_data[key][sub_key][()]
    return data


# Function to convert specified variables to CSV format
def convert_variables_to_csv(data_dict, variables, output_path):
    """
    Converts specified variables from the data dictionary to CSV files, one per sample.

    Args:
        data_dict (dict): Dictionary containing the loaded data.
        variables (list): List of variable names to convert to CSV.
        output_path (Path): Path to save the converted CSV files.
    """
    # Ensure the required variables are present
    missing_vars = [var for var in variables if var not in data_dict]
    if missing_vars:
        raise ValueError(f"The following variables are missing in the data: {missing_vars}")

    # Transpose data if needed to ensure the shape is consistent (e.g., (20, 1493))
    for var in variables:
        data_dict[var] = data_dict[var].T

    # Merge and save data for each sample
    num_samples = data_dict[variables[0]].shape[1]
    for i in range(num_samples):
        sample_data = {var: data_dict[var][:, i] for var in variables}
        df = pd.DataFrame(sample_data)
        csv_file_path = Path(output_path, f'Sample_{i + 1}.csv')
        df.to_csv(csv_file_path, index=False)
        print(f"Sample {i + 1} saved to {csv_file_path}")


# Main function to process multiple files
def process_files(filenames, variables_to_convert):
    """
    Processes a list of filenames by loading each .mat file,
    converting specified variables to individual CSV files for each sample.

    Args:
        filenames (list): List of .mat filenames without extension.
        variables_to_convert (list): List of variable names to convert to CSV.
    """
    for filename in filenames:
        # Set paths
        mat_file_path = Path(CONST.DATA_PATH, 'raw', f'{filename}.mat')
        results_path = Path(CONST.DATA_PATH, filename)

        # Prepare output directory
        prepare_output_directory(results_path)

        # Load data from .mat file
        data_dict = load_dataset(mat_file_path)

        # Display data structure for analysis
        print(f"Data structure for {filename}.mat:")
        for key, value in data_dict.items():
            print(f"{key}: {type(value)}, shape: {value.shape}")

        # Convert specified variables to CSV
        convert_variables_to_csv(data_dict, variables_to_convert, results_path)


# Example function call from a main script
if __name__ == "__main__":
    # List of .mat filenames to process (without file extensions)
    filenames = ['Tensile_Test_PBTGF0', 'Tensile_Test_PBTGF30']
    # List of variables to be converted
    variables_to_convert = ['time', 'Force', 'Strain_l_75_smooth']

    # Process each file
    process_files(filenames, variables_to_convert)