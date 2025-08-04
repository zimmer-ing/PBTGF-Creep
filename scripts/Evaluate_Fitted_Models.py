# Copyright (c) 2025 Bernd Zimmering
# This file is part of the PBTGF-Creep project (https://github.com/zimmer-ing/PBTGF-Creep).
# Licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# If you use this code or the accompanying dataset, please cite:
# Klatt, E.; Zimmering, B.; Niggemann, O.; Rauter, N.:
# "Machine-Learning-Enabled Comparative Modelling of the Creep Behaviour of Unreinforced PBT and Short-Fibre Reinforced PBT Using Prony and Fractional Derivative Models", Appl. Mech. 2025.
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import torch

import Constants as CONST
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as mtick
from src.models.FractionalDamper_Multiprocessing import FractionalDamperModel
weighting_function=FractionalDamperModel.weighting_function


def parse_prony_parameters(model_name):
    """
    Parse the model name to determine the number of Prony terms and construct the parameter list accordingly.
    Example: model_name = "Prony_3" -> parameters = ["K_eq", "K_1", "tau_1", "K_2", "tau_2", "K_3", "tau_3"]

    Parameters
    ----------
    model_name : str
        Name of the model, e.g. "Prony_2", "Prony_3".

    Returns
    -------
    list of str
        Parameter names corresponding to the given Prony model.
    """
    try:
        # Extract the number of terms from the model name
        n_terms = int(model_name.split("_")[1])
    except (IndexError, ValueError):
        raise ValueError(f"Could not parse Prony terms from model name '{model_name}'.")

    # Start with the equilibrium stiffness parameter
    parameters = ["K_eq"]
    # Add K_i and tau_i for each term
    for i in range(1, n_terms + 1):
        parameters.append(f"K_{i}")
        parameters.append(f"tau_{i}")
    return parameters


def load_parameters(path, parameter_names):
    """
    Load model parameters from a CSV file.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the CSV file containing the parameters.
    parameter_names : list of str
        Names of the parameters to load.

    Returns
    -------
    dict
        A dictionary of parameter_name: array_of_values
    """
    df_params = pd.read_csv(Path(path))
    return {param: df_params[param].values for param in parameter_names if param in df_params.columns}


def load_sample_losses(path, dataset_name):
    """
    Load sample-based losses from a CSV file.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the directory containing the loss files.
    dataset_name : str
        Name of the dataset.

    Returns
    -------
    ndarray
        Numpy array of losses.
    """
    sample_loss_file = Path(path, f"{dataset_name}_sample_losses.csv")
    if sample_loss_file.exists():
        return pd.read_csv(sample_loss_file).values
    else:
        raise FileNotFoundError(f"Sample loss file {sample_loss_file} not found.")


def generate_latex_table_parameters(parameters, dataset_name, model_name, output_dir):
    """
    Generate a LaTeX table listing the parameters for a given model and dataset.

    Parameters
    ----------
    parameters : dict
        Dictionary of parameter_name: values (one value per sample).
    dataset_name : str
        Name of the dataset.
    model_name : str
        Name of the model.
    output_dir : pathlib.Path
        Directory where the LaTeX file will be saved.
    """


    #remove underscores form names
    model_name = model_name.replace("_", " ")
    dataset_name = dataset_name.replace("_", " ")

    latex_str = "\\begin{table}[htbp]\n"
    latex_str += "\\centering\n"
    latex_str += f"\\caption{{Parameters for {model_name} - {dataset_name}}}\n"
    latex_str += "\\begin{tabular}{|c|" + "c" * len(parameters) + "|}\n"
    latex_str += "\\hline\n"
    latex_str += "Sample & " + " & ".join([latex_param_map(param) for param in parameters.keys()]) + " \\\\\n"
    latex_str += "\\hline\n"

    num_samples = len(next(iter(parameters.values())))
    for i in range(num_samples):
        row_values = " & ".join(
            f"{parameters[param][i]:{format_map.get(param, '.0f')}}"  # Default 0f
            for param in parameters.keys()
        )
        latex_str += f"{i} & {row_values} \\\\\n"

    latex_str += "\\hline\n"
    latex_str += "\\end{tabular}\n"
    latex_str += f"\\label{{tab:params_{model_name}_{dataset_name}}}\n"
    latex_str += "\\end{table}\n"

    output_file = Path(output_dir, f"{model_name}_parameters_{dataset_name}.tex")
    with open(output_file, "w") as f:
        f.write(latex_str)

def plot_predictions_norm0(time, predictions, truth, losses, dataset_name, model_name, output_dir):
    """
    Plot predictions, ground truth, and losses for each sample and save as an HTML file.

    Parameters
    ----------
    time : pd.DataFrame
        DataFrame containing time values for each sample (each column is one sample).
    predictions : pd.DataFrame
        DataFrame containing predicted values (each column is one sample).
    truth : pd.DataFrame
        DataFrame containing ground truth values (each column is one sample).
    losses : pd.DataFrame
        DataFrame containing L1 losses (each column is one sample).
    dataset_name : str
        Name of the dataset.
    model_name : str
        Name of the model.
    output_dir : pathlib.Path
        Directory where the plot will be saved.
    """
    #norm truth and predictions to 0 by substracting the first point of the truth
    inital_truth = truth.iloc[0, :]
    truth = truth - inital_truth
    predictions = predictions - inital_truth
    fig = go.Figure()
    for i in range(len(predictions.iloc[0, :])):
        x_axis = time.iloc[:, i]
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=predictions.iloc[:, i],
            mode='lines',
            name=f"Sample {i} Prediction ",
            line=dict(dash='dashdot')
        ))
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=truth.iloc[:, i],
            mode='lines',
            name=f"Sample {i} Truth "
        ))
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=losses.iloc[:, i],
            mode='lines',
            name=f"Sample {i} L1 Loss",
            yaxis='y2',
            visible='legendonly'
        ))

    fig.update_layout(
        title=f"Predictions for {model_name} - {dataset_name}",
        xaxis_title="Time",
        yaxis_title="Strain",
        yaxis2=dict(
            title="L1 Loss",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            x=1.1,
            y=1,
            xanchor='left',
            yanchor='top'
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir, f"Normed_Predictions{model_name}_{dataset_name}.html")
    fig.write_html(output_file)
    print(f"Saved predictions plot to {output_file}")


def plot_predictions(time, predictions, truth, losses, dataset_name, model_name, output_dir):
    """
    Plot predictions, ground truth, and losses for each sample and save as an HTML file.

    Parameters
    ----------
    time : pd.DataFrame
        DataFrame containing time values for each sample (each column is one sample).
    predictions : pd.DataFrame
        DataFrame containing predicted values (each column is one sample).
    truth : pd.DataFrame
        DataFrame containing ground truth values (each column is one sample).
    losses : pd.DataFrame
        DataFrame containing L1 losses (each column is one sample).
    dataset_name : str
        Name of the dataset.
    model_name : str
        Name of the model.
    output_dir : pathlib.Path
        Directory where the plot will be saved.
    """
    fig = go.Figure()
    for i in range(len(predictions.iloc[0, :])):
        x_axis = time.iloc[:, i]
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=predictions.iloc[:, i],
            mode='lines',
            name=f"Sample {i} Prediction ",
            line=dict(dash='dashdot')
        ))
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=truth.iloc[:, i],
            mode='lines',
            name=f"Sample {i} Truth "
        ))
        fig.add_trace(go.Scatter(
            x=x_axis,
            y=losses.iloc[:, i],
            mode='lines',
            name=f"Sample {i} L1 Loss",
            yaxis='y2',
            visible='legendonly'
        ))

    fig.update_layout(
        title=f"Predictions for {model_name} - {dataset_name}",
        xaxis_title="Time",
        yaxis_title="Strain",
        yaxis2=dict(
            title="L1 Loss",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            x=1.1,
            y=1,
            xanchor='left',
            yanchor='top'
        )
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(output_dir, f"Predictions{model_name}_{dataset_name}.html")
    fig.write_html(output_file)
    print(f"Saved predictions plot to {output_file}")


def create_loss_comparison_table(models, datasets, losses_dict, results_path_output):
    """
    Create a LaTeX table comparing mean L1 losses per sample for each model,
    and show the percentage difference relative to the FractionalDamper model.

    Requirements:
    - First column: "Smp."
    - FractionalDamper: "Fractional\\Damper" in one column
    - Prony models with a diff column: Two columns side by side:
      First column = "Prony i", second column = "\\%diff"
      No vertical line between these two columns, but a vertical line after them.
    - A horizontal line before the "Average" row.
    - "Average" row values in bold.

    Parameters
    ----------
    models : list of str
        List of model names.
    datasets : list of str
        List of dataset names.
    losses_dict : dict
        Nested dictionary: losses_dict[dataset][model_name] = DataFrame of L1 losses over time for each sample.
    results_path_output : pathlib.Path
        Path to the directory where evaluation outputs will be saved.
    """

    if "FractionalDamper" not in models:
        print("FractionalDamper not found in models. Cannot compute percentage differences.")
        return

    output_dir = Path(results_path_output, "latex_tables")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Display names for models
    display_names = {
        "FractionalDamper": "\makecell{Fractional \\\ Damper}",
    }
    for m in models:
        if m.startswith("Prony_"):
            parts = m.split("_")
            display_names[m] = f"Prony {parts[1]}"

    sample_col_name = "Smp."

    for dataset in datasets:
        model_data = {}
        for model_name in models:
            if (model_name in losses_dict[dataset] and
                isinstance(losses_dict[dataset][model_name], pd.DataFrame) and
                not losses_dict[dataset][model_name].empty):
                mean_losses_per_sample = losses_dict[dataset][model_name].mean(axis=0)
                model_data[model_name] = mean_losses_per_sample.values
            else:
                model_data[model_name] = None

        df_losses = pd.DataFrame({m: model_data[m] for m in model_data if model_data[m] is not None})

        if df_losses.empty:
            print(f"No loss data available for dataset {dataset}. Skipping loss comparison table.")
            continue

        # Compute %diff relative to FractionalDamper
        frac_losses = df_losses["FractionalDamper"]
        for m in df_losses.columns:
            if m != "FractionalDamper":
                df_losses[m + "_%diff"] = ((df_losses[m] - frac_losses) / frac_losses) * 100

        # Add Average row
        avg_row = df_losses.mean(axis=0)
        avg_row.name = "Average"
        df_losses = pd.concat([df_losses, avg_row.to_frame().T], axis=0)

        # Determine columns order and also build tabular specification
        column_specs = ["|c|"]  # for Smp.
        columns = [sample_col_name]

        for m in models:
            if m in df_losses.columns:
                if m == "FractionalDamper":
                    # Just one column for FDamper
                    column_specs.append("c|")
                    columns.append(m)
                else:
                    # Check if diff column exists
                    if (m + "_%diff") in df_losses.columns:
                        # Two columns for this model: model and diff
                        # No vertical line between them, but a line after
                        # So we add "c c|" to column_specs
                        column_specs.append("c c|")
                        columns.append(m)
                        columns.append(m + "_%diff")
                    else:
                        # Just one column for this model
                        column_specs.append("c|")
                        columns.append(m)

        # Join column_specs into one string
        tabular_spec = "".join(column_specs)

        df_losses_for_table = df_losses[columns[1:]]

        latex_dataset = dataset.replace("_", " ")

        latex_str = "\\begin{table}[htbp]\n"
        latex_str += "\\centering\n"
        latex_str += f"\\caption{{Mean L1 Loss Comparison for {latex_dataset}}}\n"
        latex_str += f"\\begin{{tabular}}{{{tabular_spec}}}\n"
        latex_str += "\\hline\n"

        # Build header row
        header_cells = []
        # First column is Smp.
        header_cells.append(sample_col_name)
        i = 1
        while i < len(columns):
            col = columns[i]
            if col == "FractionalDamper":
                header_cells.append(display_names.get(col, col))
                i += 1
            elif col.startswith("Prony_"):
                # Next column should be the diff column
                model_display = display_names.get(col, col)
                diff_col = columns[i+1]  # this should be m + "_%diff"
                header_cells.append(model_display)
                header_cells.append("\\%diff")
                i += 2
            else:
                # A model without diff
                header_cells.append(display_names.get(col, col.replace("_", " ")))
                i += 1

        header_row = " & ".join(header_cells) + " \\\\\n"
        latex_str += header_row
        latex_str += "\\hline\n"

        for idx, row_data in df_losses_for_table.iterrows():
            sample_label = str(idx)
            is_average = (sample_label == "Average")
            if is_average:
                latex_str += "\\hline\n"
                sample_label = "\\textbf{Average}"

            row_values = [sample_label]

            # Fill row values
            for c_name in df_losses_for_table.columns:
                val = row_data[c_name]
                is_diff = c_name.endswith("_%diff")

                if is_diff:
                    val_str = f"{val:.1f}\\%"
                else:
                    val_str = f"{val:.2e}"

                if is_average:
                    val_str = "\\textbf{" + val_str + "}"

                row_values.append(val_str)

            latex_str += " & ".join(row_values) + " \\\\\n"

        latex_str += "\\hline\n"
        latex_str += "\\end{tabular}\n"
        latex_str += f"\\label{{tab:loss_comparison_{model_name}_{dataset}}}\n"
        latex_str += "\\end{table}\n"

        output_file = Path(output_dir, f"LossComparison_{dataset}.tex")
        with open(output_file, "w") as f:
            f.write(latex_str)
        print(f"Saved loss comparison table to {output_file}")



def generate_model_wise_prony_table(parameters_dict, datasets, prony_model, output_dir):
    """
    Generate a compact LaTeX table for mean values and standard deviations of Prony parameters
    for a single model across multiple datasets, using the format 'Mean ± StdDev'.

    Parameters
    ----------
    parameters_dict : dict
        A dictionary where keys are dataset names and values are DataFrames
        containing parameter values for the specified model.
    datasets : list of str
        List of dataset names.
    prony_model : str
        Name of the Prony model (e.g., "Prony_1").
    output_dir : pathlib.Path
        Directory where the LaTeX file will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove underscores from the model name and extract the number of terms
    prony_model_clean = prony_model.replace("_", " ")
    num_terms = prony_model.split("_")[1]

    # Extract the parameters for the given model and datasets
    parameters = {dataset: parameters_dict[dataset] for dataset in datasets}



    # Begin the LaTeX table
    latex_str = "\\begin{table}[H]\n"
    latex_str += "\\centering\n"  # Center the table
    latex_str += "\\renewcommand{\\arraystretch}{0.8} % Compact table rows\n"  # Reduce vertical spacing
    latex_str += f"\\caption{{Mean values and standard deviations (Mean ± StdDev) of Prony-series coefficients for {prony_model_clean} with {num_terms} terms.}}\n"
    latex_str += f"\\label{{tab:prony_summary_{prony_model}}}\n"
    latex_str += "\\begin{tabular}{l" + "c" * len(datasets) + "}\n"  # Left align the parameter column, center datasets
    latex_str += "\\toprule\n"
    latex_str += "Parameter & " + " & ".join([f"\\textbf{{{dataset.replace('_', ' ')}}}" for dataset in datasets]) + " \\\\\n"
    latex_str += "\\midrule\n"

    # Iterate through parameters and combine mean and standard deviation
    for param in parameters[datasets[0]].keys():
        latex_param = latex_param_map(param)
        mean_std_values = [
            f"${parameters[dataset][param].mean():.2f} \\pm {parameters[dataset][param].std():.2f}$"
            for dataset in datasets
        ]
        latex_str += f"{latex_param} & " + " & ".join(mean_std_values) + " \\\\\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"

    # Save the file
    output_file = Path(output_dir, 'latex_tables', f"{prony_model}_summary_table.tex")
    with open(output_file, "w") as f:
        f.write(latex_str)
    print(f"LaTeX table saved to {output_file}")

def generate_fractional_element_table(parameters_dict, datasets, output_dir):
    """
    Generate a compact LaTeX table for mean values and standard deviations
    of Fractional Element parameters across multiple datasets, using the format 'Mean ± StdDev'.

    Parameters
    ----------
    parameters_dict : dict
        A dictionary where keys are dataset names and values are DataFrames
        containing parameter values for the Fractional Element.
    datasets : list of str
        List of dataset names.
    output_dir : pathlib.Path
        Directory where the LaTeX file will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)



    # Begin the LaTeX table
    latex_str = "\\begin{table}[H]\n"
    latex_str += "\\centering\n"  # Center the table
    latex_str += "\\renewcommand{\\arraystretch}{0.8} % Compact table rows\n"  # Reduce vertical spacing
    latex_str += "\\caption{Mean values and standard deviations (Mean ± StdDev) of Fractional Element parameters.}\n"
    latex_str += "\\label{tab:fractional_element_summary}\n"
    latex_str += "\\begin{tabular}{l" + "c" * len(datasets) + "}\n"  # Left align the parameter column, center datasets
    latex_str += "\\toprule\n"
    latex_str += "Parameter & " + " & ".join([f"\\textbf{{{dataset.replace('_', ' ')}}}" for dataset in datasets]) + " \\\\\n"
    latex_str += "\\midrule\n"

    # Iterate through parameters and combine mean and standard deviation
    for param in parameters_dict[datasets[0]].keys():
        latex_param = latex_param_map(param)
        mean_std_values = [
            f"${parameters_dict[dataset][param].mean():{format_map.get(param, '.0f')}} \\pm {parameters_dict[dataset][param].std():{format_map.get(param, '.0f')}}$"
            for dataset in datasets
        ]
        latex_str += f"{latex_param} & " + " & ".join(mean_std_values) + " \\\\\n"

    latex_str += "\\bottomrule\n"
    latex_str += "\\end{tabular}\n"
    latex_str += "\\end{table}\n"

    # Save the file
    output_file = Path(output_dir, 'latex_tables', f"fractional_element_summary_table.tex")
    with open(output_file, "w") as f:
        f.write(latex_str)
    print(f"LaTeX table saved to {output_file}")
def process_models(models, datasets, results_path_input, results_path_output):
    """
    Process each model for each dataset:
    - Load parameters, predictions, truth, and losses.
    - Generate LaTeX tables for parameters.
    - Plot predictions vs truth and losses.
    - Compute and plot mean L1 losses.

    Parameters
    ----------
    models : list of str
        List of model names (e.g. ["FractionalDamper", "Prony_2", "Prony_3"]).
    datasets : list of str
        List of dataset names.
    results_path_input : pathlib.Path
        Path to the directory where model fitting results are stored.
    results_path_output : pathlib.Path
        Path to the directory where evaluation outputs will be stored.
    """
    # Dictionaries to store results for later use (plotting mean losses)
    losses_dict = {dataset: {model: {} for model in models} for dataset in datasets}
    predictions_dict = {dataset: {model: {} for model in models} for dataset in datasets}
    truth_dict = {dataset: {model: {} for model in models} for dataset in datasets}

    # Iterate over each model
    for model_name in models:
        print(f"Processing model: {model_name}")

        # Determine if this is a Prony model and generate parameters dynamically if so
        if model_name.startswith("Prony_"):
            parameter_names = parse_prony_parameters(model_name)
        elif model_name == "FractionalDamper":
            # For FractionalDamper, we define parameters manually
            parameter_names = ['e_0', 'alpha']
        else:
            # For other models (if any), define parameters here or raise an error
            raise ValueError(f"Unknown model: {model_name}. Please define its parameters.")
        params_over_datasets = {dataset: {} for dataset in datasets}
        for dataset in datasets:
            print(f"  Processing dataset: {dataset}")
            model_results_path = Path(results_path_input, model_name)

            # Load parameters
            params_file = Path(model_results_path, f"{dataset}_params.csv")
            if not params_file.exists():
                print(f"Parameters file not found for {model_name}, {dataset}. Skipping...")
                continue
            parameters = load_parameters(params_file, parameter_names)

            # After: parameters = load_parameters(params_file, parameter_names)
            # Check if we're dealing with a Prony model:
            if model_name.startswith("Prony_"):
                # Gather all K_ columns, including "K_eq"
                k_cols = [col for col in parameters.keys() if col.startswith("K_")]

                # Sum all K_i values per sample (element-wise sum)
                k_total_values = np.zeros_like(parameters[k_cols[0]])
                for k_col in k_cols:
                    k_total_values += parameters[k_col]

                # Store it as a new entry "K_total"
                parameters["K_total"] = k_total_values
            params_over_datasets[dataset] = parameters

            # Generate LaTeX tables for parameters
            output_dir = Path(results_path_output, "latex_tables")
            output_dir.mkdir(parents=True, exist_ok=True)
            generate_latex_table_parameters(parameters, dataset, model_name, output_dir)
            print(f"Saved parameter LaTeX tables for {model_name} - {dataset}")

            # Load losses
            loss_file = Path(model_results_path, f"{dataset}_L1_loss.csv")
            if not loss_file.exists():
                print(f"Loss file not found for {model_name}, {dataset}. Skipping...")
                continue
            losses = pd.read_csv(loss_file)
            losses_dict[dataset][model_name] = losses

            # Load predictions
            pred_file = Path(model_results_path, f"{dataset}_predictions.csv")
            if not pred_file.exists():
                print(f"Predictions file not found for {model_name}, {dataset}. Skipping...")
                continue
            predictions = pd.read_csv(pred_file)
            predictions_dict[dataset][model_name] = predictions

            # Load truth
            truth_file = Path(model_results_path, f"{dataset}_truth.csv")
            if not truth_file.exists():
                print(f"Truth file not found for {model_name}, {dataset}. Skipping...")
                continue
            truth_data = pd.read_csv(truth_file)
            truth_dict[dataset][model_name] = truth_data

            # Load time
            time_file = Path(model_results_path, f"{dataset}_time.csv")
            if not time_file.exists():
                print(f"Time file not found for {model_name}, {dataset}. Skipping...")
                continue
            time_data = pd.read_csv(time_file)

            # Plot predictions and truth
            plot_dir = Path(results_path_output, "plots")
            plot_predictions(time_data, predictions, truth_data, losses, dataset, model_name, plot_dir)
            plot_predictions_norm0(time_data, predictions, truth_data, losses, dataset, model_name, plot_dir)

        if model_name.startswith("Prony_"):
            generate_model_wise_prony_table(params_over_datasets, datasets, model_name, results_path_output)
        if model_name == "FractionalDamper":
            generate_fractional_element_table(params_over_datasets, datasets, results_path_output)


    # Plot mean L1 loss for all datasets and models
    fig = go.Figure()
    for dataset in datasets:
        for model_name in models:
            if isinstance(losses_dict[dataset][model_name], pd.DataFrame) and not losses_dict[dataset][
                model_name].empty:
                mean_loss_over_samples = losses_dict[dataset][model_name].mean(axis=1)
                fig.add_trace(go.Scatter(
                    x=np.arange(len(mean_loss_over_samples)),
                    y=mean_loss_over_samples,
                    mode='lines',
                    name=f"{model_name} - {dataset}"
                ))
    fig.update_layout(
        title="Mean L1 Loss over all samples",
        xaxis_title="Epochs",
        yaxis_title="L1 Loss"
    )

    plot_dir = Path(results_path_output, "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(plot_dir, "mean_losses_all_models.html")
    fig.write_html(output_file)
    print(f"Saved mean losses plot to {output_file}")

    #loss comparison table
    create_loss_comparison_table(models, datasets, losses_dict, results_path_output)




def plot_master_curve(time, measurements, dataset_name, output_dir, predictions=None):
    """
    Plot the master curve (mean) of the truth measurements with standard deviation shading,
    and overlay the master curves of predictions from different models (without shading).

    Parameters
    ----------
    time : pd.DataFrame or pd.Series
        Time values. If a DataFrame is provided, the first column is used as the time vector.
    measurements : pd.DataFrame
        DataFrame where each column represents a truth measurement (e.g., a sample's curve).
    dataset_name : str
        Name of the dataset; used in the plot title and output filename.
    output_dir : pathlib.Path
        Directory where the HTML plot file will be saved.
    predictions : dict, optional
        Dictionary of predictions DataFrames keyed by model name. Each DataFrame should have the
        same number of rows as 'measurements'. If provided, the master curve (mean) of each model's
        predictions will be added to the plot.
    """
    # If time is a DataFrame, use its first column as the time vector
    if isinstance(time, pd.DataFrame):
        time_vals = time.iloc[:, 0]
    else:
        time_vals = time

    # Compute the master curve (mean) and standard deviation for the truth measurements
    mean_truth = measurements.mean(axis=1)
    std_truth = measurements.std(axis=1)
    upper_bound = mean_truth + std_truth
    lower_bound = mean_truth - std_truth

    # Create the Plotly figure
    fig = go.Figure()

    # Add shaded area for truth standard deviation
    fig.add_trace(go.Scatter(
        x=list(time_vals) + list(time_vals[::-1]),
        y=list(upper_bound) + list(lower_bound[::-1]),
        fill='toself',
        fillcolor='rgba(128,128,128,0.3)',  # gray with transparency
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=True,
        name="Truth Std. Dev."
    ))

    # Add the truth master curve (mean)
    fig.add_trace(go.Scatter(
        x=time_vals,
        y=mean_truth,
        mode='lines',
        line=dict(color='blue'),
        name='Master Curve (Truth)'
    ))

    # If predictions are provided, overlay their master curves (without shading)
    if predictions is not None:
        for model_name, pred_data in predictions.items():
            # Compute the mean curve for the predictions of the model
            mean_pred = pred_data.mean(axis=1)
            fig.add_trace(go.Scatter(
                x=time_vals,
                y=mean_pred,
                mode='lines',
                line=dict(dash='dash'),
                name=f'Master Curve (Prediction: {model_name})'
            ))

    # Update layout with title and axis labels
    fig.update_layout(
        title=f"Master Curve for {dataset_name} (Truth & Predictions)",
        xaxis_title="Time",
        yaxis_title="Measurement Value"
    )

    # Ensure output directory exists and save the plot as HTML
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset_name}_master_curve.html"
    fig.write_html(str(output_file))
    print(f"Master Curve plot saved to: {output_file}")


def plot_master_curve_latex(time, truth, dataset, output_dir, predictions=None, rename_dict=None, linestyles=None, figsize=(394/72, 222/72), linewidth=1):
    """
    Plot a LaTeX-styled master curve (truth data with standard deviation shading)
    and overlay the master curves of predictions from different models with specified linestyles.
    The plot is saved as a PDF with fixed dimensions.

    Parameters
    ----------
    time : pd.DataFrame or pd.Series
        Time values. If a DataFrame is provided, the first column is used.
    truth : pd.DataFrame
        Truth measurements; each column represents a sample.
    dataset : str
        Name of the dataset (used in title/labels).
    output_dir : pathlib.Path
        Directory where the PDF plot will be saved.
    predictions : dict, optional
        Dictionary of predictions (key: model name, value: DataFrame with the same dimensions as truth).
    rename_dict : dict, optional
        Dictionary mapping model names to new labels.
    linestyles : dict, optional
        Dictionary mapping model names to desired linestyles (e.g., '-', '--', '-.', ':').
    figsize : tuple, optional
        Figure size in inches (default is (8,6)).
    linewidth : float, optional
        Line width for the plotted curves.
    """

    mpl.rc('text', usetex=True)
    mpl.rc('font', family='serif')

    if hasattr(time, 'iloc'):
        time_vals = time.iloc[:, 0]
    else:
        time_vals = time

    mean_truth = truth.mean(axis=1)
    std_truth = truth.std(axis=1)
    upper_bound = mean_truth + std_truth
    lower_bound = mean_truth - std_truth

    # Determine dynamic y-axis limits with 10% margin
    y_min = lower_bound.min()
    y_max = upper_bound.max()
    y_range = y_max - y_min
    y_lim_lower = y_min - 0.02 * y_range
    y_lim_upper = y_max + 0.02 * y_range

    fig, ax = plt.subplots(figsize=figsize)

    ax.fill_between(time_vals, lower_bound, upper_bound, color='gray', alpha=0.3,
                    label='Standard Deviation')

    ax.plot(time_vals, mean_truth, color='blue', linewidth=linewidth, label='Mastercurve (Truth)')

    if predictions is not None:
        for model_name, pred_data in predictions.items():
            mean_pred = pred_data.mean(axis=1)
            if rename_dict is not None and model_name in rename_dict:
                label = rename_dict[model_name]
            else:
                label = model_name.replace('_', ' ')
            if linestyles is not None and model_name in linestyles:
                ls = linestyles[model_name]
            else:
                ls = '--'
            ax.plot(time_vals, mean_pred, linestyle=ls, linewidth=linewidth, label=label)

    ax.set_xlabel(r'Time $t$ [s]', fontsize=10)
    ax.set_ylabel(r'Strain $\varepsilon$ [\%]', fontsize=10)
    #ax.set_title('Mastercurve - ' + dataset.replace('_', ' '), fontsize=14)
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(False)

    ax.set_xlim([time_vals.min(), time_vals.max()])
    ax.set_ylim([y_lim_lower, y_lim_upper])
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.3f'))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{dataset}_mastercurve_latex.pdf"
    plt.tight_layout()
    plt.savefig(output_file, format='pdf')
    plt.close(fig)
    print(f"LaTeX-styled master curve plot saved to: {output_file}")



def plot_loss_weighting_with_mastercurve(time, master_curve, results_path, dataset, figsize=(394/72, 222/72)):
    """
    Plot the loss weighting function based on time with a second y-axis for the master curve.

    Parameters
    ----------
    time : np.ndarray
        Array of time values.
    master_curve : np.ndarray
        Werte der Master Curve.
    results_path : str
        Pfad, in dem die PDF gespeichert werden soll.
    dataset : str
        Name des Datensatzes (wird im Label der Master Curve verwendet).
    figsize : tuple, optional
        Figure size in inches, default is (394/72, 222/72).
    """
    # Berechne die Loss-Gewichtung für jeden Zeitwert
    weights = weighting_function(torch.tensor(time), alpha=10.0, tau=15, baseline=1.0).detach().numpy()

    # Erstelle den Plot mit der Haupt-Y-Achse (für die Loss-Gewichtung)
    fig, ax1 = plt.subplots(figsize=figsize)
    ax1.plot(time, weights, label='Loss Weighting', color='orange')
    ax1.set_xlabel(r'$\mathrm{Time\ [s]}$', fontsize=10, fontweight='normal')
    ax1.set_ylabel(r'$\mathrm{Weighting\ Factor}$', fontsize=10, fontweight='normal')
    ax1.tick_params(axis='both', which='major', labelsize=10, colors='black')

    # Erstelle eine zweite Y-Achse für die Master Curve
    ax2 = ax1.twinx()
    ax2.plot(time, master_curve, label=f'Master Curve {dataset}', color='blue')
    ax2.set_ylabel(r'$\mathrm{Master\ Curve}$', fontsize=10, fontweight='normal')
    ax2.tick_params(axis='y', labelsize=10, colors='black')

    # Kombinierte Legende aus beiden Achsen, positioniert rechts im Plot (innerhalb)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)

    # Speichern des Plots als PDF
    pdf_filename = Path(results_path, f"loss_weighting_{dataset}.pdf")
    fig.savefig(str(pdf_filename), format='pdf', bbox_inches='tight')
    plt.close(fig)


def process_master_curves(models, datasets, results_path_input, results_path_output):
    """
    Process each dataset to generate master curve plots showing the truth data (with standard deviation shading)
    and the master curves of predictions from each model.

    For each dataset, this function:
    - Loads the time vector and truth measurements from CSV files in a designated folder (using the first model's folder).
    - Iterates over all models to load their predictions from their respective folders.
    - Calls both plot_master_curve() (Plotly HTML version) and plot_master_curve_latex() (matplotlib PDF version).

    Parameters
    ----------
    models : list of str
        List of model names. The first model is assumed to contain the truth data.
    datasets : list of str
        List of dataset names.
    results_path_input : pathlib.Path
        Path to the directory where the measurement files are stored.
    results_path_output : pathlib.Path
        Path to the directory where the master curve plots will be saved.
    """
    # Create an output subdirectory for master curves
    master_curve_output_dir = Path(results_path_output , "master_curves")
    master_curve_output_dir.mkdir(parents=True, exist_ok=True)

    weighting_curve_output_dir = Path(results_path_output , "loss_weighting")
    weighting_curve_output_dir.mkdir(parents=True, exist_ok=True)

    # Assume truth data is in the folder for the first model
    truth_folder = models[0]

    for dataset in datasets:
        print(f"Processing master curve for dataset: {dataset}")

        # Construct file paths for time and truth measurements
        time_file = Path(results_path_input, truth_folder, f"{dataset}_time.csv")
        truth_file = Path(results_path_input, truth_folder, f"{dataset}_truth.csv")

        if not time_file.exists():
            print(f"Time file not found for dataset {dataset}: {time_file}. Skipping...")
            continue
        if not truth_file.exists():
            print(f"Truth file not found for dataset {dataset}: {truth_file}. Skipping...")
            continue

        # Load time vector and truth measurements
        time_data = pd.read_csv(str(time_file))
        truth_data = pd.read_csv(str(truth_file))

        # Build a dictionary of predictions for each model
        predictions = {}
        for model in models:
            pred_file = Path(results_path_input, model, f"{dataset}_predictions.csv")
            if pred_file.exists():
                pred_data = pd.read_csv(str(pred_file))
                predictions[model] = pred_data
            else:
                print(
                    f"Predictions file not found for model {model}, dataset {dataset}. Skipping predictions for this model.")

        # Call the Plotly-based master curve function
        plot_master_curve(time_data, truth_data, dataset, master_curve_output_dir, predictions)

        # Call the new LaTeX-styled master curve function using matplotlib
        plot_master_curve_latex(time_data, truth_data, dataset, master_curve_output_dir, predictions,rename_dict,linestyles)

        truth_mean = truth_data.mean(axis=1)
        plot_loss_weighting_with_mastercurve(time_data.iloc[:,0],truth_mean, weighting_curve_output_dir,dataset)


 # Map parameter names to LaTeX-friendly versions
def latex_param_map(param):
    if param == "e_0":
        return "$C$"
    elif param == "alpha":
        return r"$\alpha$"
    elif "K_eq" in param:
        return "$K_{eq}$"
    if param == "K_total":
        return r"$K_{\mathrm{total}}$"
    elif "tau" in param:
        return f"$\\tau_{{{param.split('_')[1]}}}$"
    elif "K" in param:
        return f"$K_{{{param.split('_')[1]}}}$"
    return param


if __name__ == "__main__":
    # Define models. Simply list them, and Prony models will be handled dynamically.
    Models = [
        'FractionalDamper',
        'Prony_1',
        'Prony_2',
        'Prony_3'
    ]
    format_map = {
        "alpha": ".2f",
        "e_0": ".2e",
    }

    rename_dict = {
        'FractionalDamper': 'Fractional Damper',
        'Prony_1': 'Prony $N=1$',
        'Prony_2': 'Prony $N=2$',
        'Prony_3': 'Prony $N=3$'
    }
    linestyles = {
        'FractionalDamper': (0, (3, 1, 1, 1)),  # custom dash pattern
        'Prony_1': '--',  # dashed
        'Prony_2': '-.',  # dash-dot
        'Prony_3': ':'  # dotted
    }

    # Define datasets
    Datasets = ['PBTGF0', 'PBTGF30']

    # Input and output paths
    #Results_Path_Input = Path(CONST.RESULTS_PATH, "fit_parameters")
    Results_Path_Input = Path(CONST.RESULTS_PATH, "Fit_models")
    Results_Path_Output = Path(CONST.RESULTS_PATH, "evaluation")

    # Process models
    process_models(Models, Datasets, Results_Path_Input, Results_Path_Output)
    process_master_curves(Models, Datasets, Results_Path_Input, Results_Path_Output)

