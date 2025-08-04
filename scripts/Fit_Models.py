# Copyright (c) 2025 Bernd Zimmering
# This file is part of the PBTGF-Creep project (https://github.com/zimmer-ing/PBTGF-Creep).
# Licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# If you use this code or the accompanying dataset, please cite:
# Klatt, E.; Zimmering, B.; Niggemann, O.; Rauter, N.:
# "Machine-Learning-Enabled Comparative Modelling of the Creep Behaviour of Unreinforced PBT and Short-Fibre Reinforced PBT Using Prony and Fractional Derivative Models", Appl. Mech. 2025.

from pathlib import Path
import os
import sys
import torch
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from sympy.abc import alpha

# Add the project source directory to the system path
PROJECT_PATH = Path(os.path.abspath(__file__)).parent.parent
sys.path.append(str(Path(PROJECT_PATH, 'src')))

from src.datasets import PBTGF0, PBTGF30
from src.models import FractionalDamper, PronyMaxwell
import Constants as CONST


def fit_models(datasets, models_to_fit):
    """
    Fit the given models to the specified datasets.

    The function supports fitting the following models:
    - Prony Maxwell models with an arbitrary number of terms (defined by model name "Prony_n").
    - Fractional Damper models.

    Parameters
    ----------
    datasets : dict
        Dictionary of dataset_name : dataset_object
    models_to_fit : list
        List of model names to fit, e.g. ["Prony_1", "Prony_2", "FractionalDamper"]
    """
    # Prepare a results directory based on the current script name
    experiment_name = Path(__file__).stem
    results_base_path = Path(CONST.RESULTS_PATH, experiment_name)
    if not results_base_path.exists():
        results_base_path.mkdir(parents=True)

    # Iterate over each dataset
    for dataset_name, dataset in datasets.items():
        print(f"Processing dataset: {dataset_name}")

        # Load and prepare the dataset
        dataset.load_data()
        dataset.prepare_data()
        strain_tensor, time_tensor, force_tensor = dataset.get_data()
        area = dataset.get_area()

        # Compute initial guess for the Fractional Damper model
        stress = force_tensor[0, 0] / area
        initial_e_0 = stress / (strain_tensor[0, 0] / 100)

        # Handle all Prony Maxwell models specified in models_to_fit
        # For example, if "Prony_2" is in models_to_fit, it will parse out n_terms=2
        prony_models = [m for m in models_to_fit if m.startswith("Prony_")]
        for prony_model_name in prony_models:
            # Parse the number of terms from the model name "Prony_n"
            try:
                n_terms = int(prony_model_name.split("_")[1])
            except (IndexError, ValueError):
                print(f"Warning: Could not parse terms from model name {prony_model_name}. Skipping...")
                continue

            print(f"Fitting Prony Model with {n_terms} term(s) for {dataset_name}...")

            # Initialize and fit the Prony Maxwell model
            prony_model = PronyMaxwell(n_terms=n_terms, area=area,
                                       results_path=Path(results_base_path, prony_model_name))
            prony_params = prony_model.fit(time_tensor, strain_tensor, force_tensor)

            # Save fitted parameters
            prony_params_df = pd.DataFrame(prony_params)
            params_file = Path(results_base_path, prony_model_name, f"{dataset_name}_params.csv")
            if not params_file.parent.exists():
                params_file.parent.mkdir(parents=True, exist_ok=True)
            prony_params_df.to_csv(params_file, index=False)

        # Handle Fractional Damper model if requested
        if "FractionalDamper" in models_to_fit:
            print(f"Fitting Fractional Damper Model for {dataset_name}...")
            fractional_damper = FractionalDamper(
                area=area,
                results_path=Path(results_base_path, "FractionalDamper"),
                initial_e_0=initial_e_0*150,
                lr_global=0.5,  # Learning rate for global optimization
                lr_finetune=0.01,  # Learning rate for fine-tuning
            )

            # Fit the Fractional Damper model
            fractional_damper_params, losses = fractional_damper.fit(time_tensor, strain_tensor, force_tensor,
                                                                     epochs_global=15, epochs_finetune=250)

            # Save fitted parameters
            fractional_damper_params_df = pd.DataFrame(fractional_damper_params)
            fd_params_file = Path(results_base_path, "FractionalDamper", f"{dataset_name}_params.csv")
            if not fd_params_file.parent.exists():
                fd_params_file.parent.mkdir(parents=True, exist_ok=True)
            fractional_damper_params_df.to_csv(fd_params_file, index=False)

            # Save losses
            losses_df = pd.DataFrame(losses)
            fd_losses_file = Path(results_base_path, "FractionalDamper", f"{dataset_name}_losses.csv")
            if not fd_losses_file.parent.exists():
                fd_losses_file.parent.mkdir(parents=True, exist_ok=True)
            losses_df.to_csv(fd_losses_file, index=False)


def plot_losses_fractional(ds_names, path_results):
    """
    For each dataset, load the Fractional Damper losses, separate global and sample-based losses,
    save them as CSV files, and plot the sample-based losses.

    Parameters
    ----------
    ds_names : iterable
        Iterable of dataset names (keys of the datasets dictionary)
    path_results : Path
        Path to the results directory where Fractional Damper results are stored.
    """
    for ds_name in ds_names:
        loss_file = Path(path_results, "FractionalDamper", f"{ds_name}_losses.csv")
        if not loss_file.exists():
            print(f"File {loss_file} does not exist.")
            continue

        losses_df = pd.read_csv(loss_file)

        # Separate global losses and sample-based losses
        global_losses = []
        sample_losses = []

        for _, row in losses_df.iterrows():
            value = row.iloc[0]  # The relevant data is in the first column
            try:
                parsed_value = eval(value) if isinstance(value, str) else value
                if isinstance(parsed_value, list):
                    sample_losses.append(parsed_value)
                else:
                    global_losses.append(parsed_value)
            except (SyntaxError, ValueError, TypeError):
                # If eval fails or value is not a valid literal, treat it as a global loss
                global_losses.append(value)

        # Convert to DataFrames
        df_losses_global = pd.DataFrame(global_losses, columns=["global_loss"])
        df_losses_sample = pd.DataFrame(sample_losses).T  # Transpose so that each sample is a column

        # Save global losses
        global_loss_file = Path(path_results, "FractionalDamper", f"{ds_name}_global_losses.csv")
        df_losses_global.to_csv(global_loss_file, index=False)

        # Save sample-based losses
        sample_loss_file = Path(path_results, "FractionalDamper", f"{ds_name}_sample_losses.csv")
        df_losses_sample.to_csv(sample_loss_file, index=False)

        # Plot the sample-based losses
        fig = px.line(df_losses_sample, title=f"Sample Losses for {ds_name}")
        fig.update_xaxes(title_text="Epoch")
        fig.update_yaxes(title_text="MSE Loss", type="log")  # Use a log axis for y
        eval_path = Path(path_results, 'evaluation')
        if not eval_path.exists():
            eval_path.mkdir(parents=True, exist_ok=True)

        fig_file = Path(eval_path, f"{ds_name}_losses.html")
        fig.write_html(fig_file)


def evaluate_training(datasets, models_to_predict):
    """
    Evaluate the trained models by:
    1. Plotting the losses for the Fractional Damper model.
    2. Predicting strain responses with the fitted parameters for each requested model.
    3. Saving predictions, ground truth, and L1 losses.

    Parameters
    ----------
    datasets : dict
        Dictionary of dataset_name : dataset_object
    models_to_predict : list
        List of model names to run predictions for, e.g. ["Prony_2", "FractionalDamper"]
    """
    experiment_name = Path(__file__).stem
    path_results = Path(CONST.RESULTS_PATH, experiment_name)
    ds_names = datasets.keys()

    # Plot losses for Fractional Damper models if any are present
    if "FractionalDamper" in models_to_predict:
        plot_losses_fractional(ds_names, path_results)

    # Predict strain using the fitted parameters for each model
    for dataset_name, dataset in datasets.items():
        dataset.load_data()
        dataset.prepare_data()
        strain, time, force = dataset.get_data()

        # Convert data to torch tensors for prediction (if needed)
        strain_tensor = torch.tensor(strain, dtype=torch.float64)
        time_tensor = torch.tensor(time, dtype=torch.float64)
        force_tensor = torch.tensor(force, dtype=torch.float64)
        area = dataset.get_area()

        # Predict with Fractional Damper if requested
        if "FractionalDamper" in models_to_predict:
            fd_params_file = Path(path_results, "FractionalDamper", f"{dataset_name}_params.csv")
            if fd_params_file.exists():
                fractional_damper_params_df = pd.read_csv(fd_params_file)
                fractional_damper = FractionalDamper(area=area,
                                                     results_path=Path(path_results, "FractionalDamper"))
                with torch.no_grad():
                    predictions = fractional_damper.predict(time_tensor, strain_tensor, force_tensor,
                                                            fractional_damper_params_df)

                # Save predictions and evaluation metrics
                predictions_df = pd.DataFrame(predictions).T
                predictions_df.to_csv(Path(path_results, "FractionalDamper", f"{dataset_name}_predictions.csv"),
                                      index=False)

                strain_df = pd.DataFrame(strain).T
                strain_df.to_csv(Path(path_results, "FractionalDamper", f"{dataset_name}_truth.csv"), index=False)

                loss = (predictions_df - strain_df).abs()
                loss.to_csv(Path(path_results, "FractionalDamper", f"{dataset_name}_L1_loss.csv"), index=False)

                time_df = pd.DataFrame(time).T
                time_df.to_csv(Path(path_results, "FractionalDamper", f"{dataset_name}_time.csv"), index=False)
            else:
                print(
                    f"File {fd_params_file} does not exist for {dataset_name}. Skipping FractionalDamper predictions...")

        # Predict with any Prony Maxwell models requested (e.g., Prony_1, Prony_2, ...)
        prony_models = [m for m in models_to_predict if m.startswith("Prony_")]
        for prony_model_name in prony_models:
            prony_params_file = Path(path_results, prony_model_name, f"{dataset_name}_params.csv")
            if not prony_params_file.exists():
                print(f"File {prony_params_file} does not exist. Skipping {prony_model_name} predictions...")
                continue

            # Parse the number of terms from the model name "Prony_n"
            try:
                n_terms = int(prony_model_name.split("_")[1])
            except (IndexError, ValueError):
                print(f"Warning: Could not parse terms from model name {prony_model_name}. Skipping...")
                continue

            prony_params_df = pd.read_csv(prony_params_file)
            prony_model = PronyMaxwell(n_terms=n_terms, area=area, results_path=Path(path_results, prony_model_name))
            predictions = prony_model.predict(time, force, prony_params_df)

            # Save predictions and evaluation metrics
            predictions_df = pd.DataFrame(predictions).T
            predictions_df.to_csv(Path(path_results, prony_model_name, f"{dataset_name}_predictions.csv"), index=False)

            strain_df = pd.DataFrame(strain).T
            strain_df.to_csv(Path(path_results, prony_model_name, f"{dataset_name}_truth.csv"), index=False)

            loss = (predictions_df - strain_df).abs()
            loss.to_csv(Path(path_results, prony_model_name, f"{dataset_name}_L1_loss.csv"), index=False)

            time_df = pd.DataFrame(time).T
            time_df.to_csv(Path(path_results, prony_model_name, f"{dataset_name}_time.csv"), index=False)


if __name__ == "__main__":
    # Load the available datasets
    datasets = {
        "PBTGF0": PBTGF0(),
        "PBTGF30": PBTGF30()
    }

    # Specify the models to process
    # You can now specify any Prony model as "Prony_n" where n is the number of terms
    #models_to_process = [ "Prony_1", "Prony_2", "Prony_3","FractionalDamper"]
    models_to_process = ["FractionalDamper"]
    #models_to_process = ["Prony_1","Prony_2", "Prony_3"]

    fit_models(datasets, models_to_process)
    evaluate_training(datasets, models_to_process)