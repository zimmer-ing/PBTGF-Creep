# Copyright (c) 2025 Bernd Zimmering
# This file is part of the PBTGF-Creep project (https://github.com/zimmer-ing/PBTGF-Creep).
# Licensed under CC BY-NC-SA 4.0: https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# If you use this code or the accompanying dataset, please cite:
# Klatt, E.; Zimmering, B.; Niggemann, O.; Rauter, N.:
# "Machine-Learning-Enabled Comparative Modelling of the Creep Behaviour of Unreinforced PBT and Short-Fibre Reinforced PBT Using Prony and Fractional Derivative Models", Appl. Mech. 2025.
import torch
from src.utils.FixedStepFDEint import FDEint
from pathlib import Path
import pandas as pd

from src.models.models_base import BaseModel
import copy
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()
seed = 42
# Set the seed for CPU
torch.manual_seed(seed)

# If you're using CUDA, set the seed for the current GPU and all GPUs
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# For reproducibility, you might also want to disable some non-deterministic behaviors:
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class FractionalElement(torch.nn.Module):
    def __init__(self, e_0, force, area):
        super(FractionalElement, self).__init__()
        if e_0 <= 0:
            raise ValueError("Initial modulus 'e_0' must be positive.")
        if force <= 0:
            raise ValueError("Force must be positive.")

        self.log_param_e_0 = torch.nn.Parameter(torch.log(e_0.clone().detach().double()))
        self.force = force
        self.area = area

    def forward(self, t, x):
        param_e_0 = torch.exp(self.log_param_e_0)
        stress = self.force / self.area
        strain_rate = (stress / param_e_0)
        return strain_rate


class FractionalDamperModel(BaseModel):
    def __init__(self, results_path, area, initial_e_0=4000, initial_alpha=0.3, lr_global=0.1, lr_finetune=0.01):
        super().__init__(None, results_path, area)
        self.initial_e_0 = torch.tensor(initial_e_0)
        self.initial_alpha = torch.tensor(initial_alpha)
        self.lr_global = lr_global
        self.lr_finetune = lr_finetune


    def loss_fn(self, y_pred, y_true,time):
        residuals = y_pred - y_true
        weights = self.weighting_function(time, alpha=10.0, tau=15, baseline=1.0)
        return torch.sum((weights * residuals) ** 2)
    @staticmethod
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
        weights = alpha * torch.exp(-time_array / tau) + baseline
        return weights

    def initialize_model(self, force):
        alpha_param = torch.nn.Parameter(self.inverse_sigmoid(self.initial_alpha))
        fractional_element = FractionalElement(e_0=self.initial_e_0, force=force, area=self.area).double()
        return fractional_element, alpha_param

    @staticmethod
    def inverse_sigmoid(alpha):
        return torch.log(alpha / (1 - alpha))

    def fit_sample(self, time_tensor, strain_tensor, fractional_element, alpha_param, epochs=100):
        s_0 = strain_tensor[0]/100
        h = time_tensor.diff(dim=0).abs().min().double()

        optimizer = torch.optim.Adam([alpha_param] + list(fractional_element.parameters()), lr=self.lr_finetune)
        loss_fn = self.loss_fn#torch.nn.functional.mse_loss
        losses = []

        with tqdm(range(epochs), desc="Fine-Tuning Epochs", dynamic_ncols=True) as epoch_pbar:
            for epoch in epoch_pbar:
                predictions = FDEint(
                    fractional_element,
                    time_tensor.unsqueeze(0),
                    s_0.unsqueeze(0),
                    torch.sigmoid(alpha_param),
                    h,
                    dtype=torch.float64,
                    DEBUG=False,
                )*100
                loss = loss_fn(predictions.squeeze(), strain_tensor.squeeze(), time_tensor)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                epoch_pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.2e}")
                losses.append(loss.item())

        trained_alpha = torch.sigmoid(alpha_param).item()
        trained_e_0 = torch.exp(fractional_element.log_param_e_0).item()

        return {"alpha": trained_alpha, "e_0": trained_e_0}, losses

    def fit(self, time_tensor, strain_tensor, force_tensor, epochs_global=100, epochs_finetune=100):
        time_tensor = torch.tensor(time_tensor)
        strain_tensor = torch.tensor(strain_tensor)
        force_tensor = torch.tensor(force_tensor)

        force = force_tensor.mean().item()
        fractional_element, alpha_param = self.initialize_model(force)

        s_0 = strain_tensor[:, 0].unsqueeze(1)/100
        h = time_tensor.diff(dim=1).abs().min().double()
        losses = []

        print("Joint optimization of e_0 and alpha...")
        optimizer_joint = torch.optim.LBFGS([alpha_param] + list(fractional_element.parameters()), lr=self.lr_global)
        with tqdm(range(epochs_global), desc="Joint Optimization", dynamic_ncols=True) as pbar:
            for epoch in pbar:
                def closure():
                    optimizer_joint.zero_grad()
                    predictions = FDEint(
                        fractional_element,
                        time_tensor,
                        s_0,
                        torch.sigmoid(alpha_param),
                        h,
                        dtype=torch.float64,
                        DEBUG=False,
                    )*100
                    loss = self.loss_fn(predictions.squeeze(-1), strain_tensor, time_tensor)
                    loss.backward()
                    return loss

                loss = optimizer_joint.step(closure)
                losses.append(loss.item())
                pbar.set_description(f"Epoch {epoch} Loss: {loss.item():.2e} alpha={torch.sigmoid(alpha_param).item()} e_0={torch.exp(fractional_element.log_param_e_0).item()}")

        print(f"Optimized parameters: alpha={torch.sigmoid(alpha_param).item()}, e_0={torch.exp(fractional_element.log_param_e_0).item()}")

        print("Fine-tuning per sample...")
        n_jobs = min(time_tensor.shape[0], num_cores)
        tasks = [
            (time_tensor[sample_idx], strain_tensor[sample_idx],
             copy.deepcopy(fractional_element), copy.deepcopy(alpha_param), epochs_finetune)
            for sample_idx in range(time_tensor.shape[0])
        ]

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(self.fit_sample)(*task) for task in tasks
        )

        all_params = []
        for sample_params, loss_sample in results:
            all_params.append(sample_params)
            losses.append(loss_sample)
            print(f"Sample optimized parameters: alpha={sample_params['alpha']}, e_0={sample_params['e_0']}")

        return all_params, losses

    def predict(self, time_tensor, strain_tensor, force_tensor, param_df):
        predictions = []
        for sample_idx, params in enumerate(param_df.to_dict(orient="records")):
            fractional_element = FractionalElement(
                e_0=torch.tensor(params["e_0"], dtype=torch.float64),
                force=force_tensor[sample_idx, :].mean().item(),
                area=self.area
            )
            alpha_param = torch.tensor(params["alpha"], dtype=torch.float64)

            time_sample = time_tensor[sample_idx]
            strain_sample = strain_tensor[sample_idx]
            s_0 = strain_sample[0].unsqueeze(0)/100
            h = time_sample.diff(dim=0).abs().min().double()

            prediction = FDEint(
                fractional_element,
                time_sample.unsqueeze(0),
                s_0,
                alpha_param,
                h,
                dtype=torch.float64,
                DEBUG=False,
            )
            predictions.append(prediction.squeeze().detach().numpy()*100)

        return predictions