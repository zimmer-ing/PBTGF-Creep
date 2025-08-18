# Project Overview

This project accompanies the publication  
**"Machine-Learning-Enabled Comparative Modelling of the Creep Behaviour of Unreinforced PBT and Short-Fibre Reinforced PBT Using Prony and Fractional Derivative Models"**  
by Eduard Klatt, Bernd Zimmering, Oliver Niggemann, and Natalie Rauter, published in *Applied Mechanics* 2025.


**The publication is open access and can be found here: [https://doi.org/10.3390/applmech6030060](https://doi.org/10.3390/applmech6030060)**

*If you use this code or dataset in academic or professional work, please cite the publication. Find the bibtex in the [last section](#Citation) of this readme.*

---

## Abstract

This study presents an approach based on data-driven methods for determining the parameters needed to model time-dependent material behaviour. The time-dependent behaviour of the thermoplastic polymer polybutylene terephthalate (PBT) is investigated, both with and without the inclusion of reinforcing short fibres. Two modelling approaches are proposed:  
- the generalised Maxwell model formulated through the classical exponential Prony series,  
- and a model based on fractional calculus.

A machine-learning-ready parameter identification framework enables the automated extraction of model parameters directly from time-series data. The fractional model leverages a novel neural solver for fractional differential equations, reducing computational complexity and allowing physical interpretation of the model parameters. Experimental data from tensile creep tests are analysed, and both predictive performance and parameter efficiency of the models are compared.  
All code and data necessary to reproduce the results of this study are available in this repository.

---

## Directory Structure

- `scripts/`: Contains Python scripts for model evaluation and analysis. The files can be executed from the command line by running `python scripts/<script_name>.py`.
    - `Data_Processing.py`: Contains functions for data processing.
    - `Fit_Models.py`: Contains functions for fitting all models to the dataset.
    - `Evaluate_Fitted_Models.py`: Script for postprocessing and evaluating the fitted models. It creates LaTeX tables and plots for the model parameters and predictions.
- `results/`: Directory where model fitting and evaluation results are stored.
- `src/`: Contains the source code for the project.
  - `datasets/`: Contains the dataset classes used in the project.
  - `models/`: Contains the models used in the project.
  - `utils/`: Contains utility functions used in the project.
- `Constants.py`: Contains constants used in the project, such as paths and filenames.

---
## General Usage Information
The commands in the following section use the Command Line Interface (CLI) to run the scripts. Ensure you have the necessary permissions to execute these commands in your terminal or command prompt.
Before running the scripts you have to setup a Python environment as described in the next section. This Python environment contains all necessary packages to run the scripts. The used packages are listed in the `requirements.txt` file as well as in following section.
All used packages are open source available on the Python Package Index (PyPI) and can be installed using `pip`. For details on the licenses of the used packages, please refer to their respective documentation.

**Science lives from acknowledgements. If you use this code or dataset in academic or professional work, please cite the publication mentioned in the last section of this manual.**


## Setting Up the Python Environment
To simplify the setup process, we recommend using **Miniforge** to create a Python environment and installing all required packages with **pip**. This approach ensures flexibility and ease of use.

### Step 1: Install Miniforge
1. Download and install Miniforge from the official [Miniforge GitHub page](https://github.com/conda-forge/miniforge).
2. Choose the appropriate installer for your operating system (e.g., `.sh` for Linux/macOS or `.exe` for Windows).
3. Follow the installation instructions provided on the GitHub page.

### Step 2: Create a Python Environment
Create a dedicated environment for the project using Python 3.11:
```bash
conda create -n PBTGF_Creep python=3.11
```

### Step 3: Activate the Environment
```bash
conda activate PBTGF_Creep
```

### Step 4: Install Dependencies
After activating the environment, install the required packages using **pip**:
```bash
pip install torch==2.5.1
pip install pandas==2.2.3
pip install matplotlib==3.9.2
pip install plotly==5.24.1
pip install scipy==1.14.1
pip install tqdm==4.62.3
pip install joblib==1.4.2
pip install h5py==3.12.1
pip install FDEint==0.1.1
```

### Step 5: Verify Installation
```bash
python -c "import torch, pandas, matplotlib, plotly, scipy, tqdm, joblib, h5py, FDEint"
```
If there are no errors, the environment is successfully set up.

### Step 6: Loading an Existing Environment
If you have already created the environment, load it with:
```bash
conda activate PBTGF_Creep
```

Now you can run the scripts in the project.

---

## Scripts

The following scripts are available for model evaluation and analysis. Ensure you have the necessary data files in the correct directories before running them.
1. **Data_Processing.py**: Converts experimental data from `.mat` files into CSV format.
2. **Fit_Models.py**: Fits all specified models to the prepared datasets.
3. **Evaluate_Fitted_Models.py**: Processes and evaluates the models for various datasets, generating LaTeX tables and plots.

### Preliminary Steps
1. navigate to the project directory:
   ```bash
   cd path/to/your/project
   ```
2. ensure the environment is activated:
   ```bash
   conda activate PBTGF_Creep
   ``` 


### Data_Processing.py
This script converts experimental data from `.mat` files into CSV format, enabling easier analysis. It extracts specified variables (e.g., `time`, `Force`, `Strain_l_75_smooth`) and saves each sample as a separate CSV file in an organized directory structure.

**Usage:**
1. Place `.mat` files in the `raw` folder under `DATA_PATH` (defined in the `Constants` file).
2. Specify filenames and variables to extract in the script.
3. Run the script:
   ```bash
   python scripts/Data_Processing.py
   ```

### Fit_Models.py
Fits all specified models to the prepared datasets.
You can run the by executing the following command:
```bash
python scripts/Fit_Models.py
```


### Evaluate_Fitted_Models.py
This script processes and evaluates the models for various datasets. It:
1. **Loads Parameters:** Loads the fitted model parameters from CSV files.
2. **Generates LaTeX Tables:** Creates LaTeX tables for the model parameters.
3. **Plots Predictions and Losses:** Creates plots for the model predictions compared to ground truth and for the loss curves.
4. **Computes and Plots Mean L1 Losses:** Computes and plots the mean L1 losses for all models and datasets.

**Example call:**
```bash
python scripts/Evaluate_fitted_models.py
```

#### Inside the script:
```python
if __name__ == "__main__":
    Models = ['FractionalDamper', 'Prony_1', 'Prony_2', 'Prony_3']
    Datasets = ['PBTGF0', 'PBTGF30']
    Results_Path_Input = Path(CONST.RESULTS_PATH, "Fit_models")
    Results_Path_Output = Path(CONST.RESULTS_PATH, "evaluation")
    process_models(Models, Datasets, Results_Path_Input, Results_Path_Output)
```

---

## Results

The results of the model evaluation are stored in the following directories:

- `results/Fit_models/`: Results of the fitted models, including the model parameters and predictions.
- `results/evaluation/latex_tables/`: Contains the generated LaTeX tables for the model parameters.
- `results/evaluation/plots/`: Contains the plots of model predictions and losses as well as the mean L1 losses.

---

## Contact
For questions about the project or setup, please contact the project owner:
- **Bernd Zimmering**  
  [bernd.zimmering@hsu-hh.de](mailto:bernd.zimmering@hsu-hh.de)   
  or open an issue on GitHub.

---

## Citation

If you use this code or dataset in academic or professional work, please cite:

```bibtex
@Article{applmech6030060,
AUTHOR = {Klatt, Eduard and Zimmering, Bernd and Niggemann, Oliver and Rauter, Natalie},
TITLE = {Machine-Learning-Enabled Comparative Modelling of the Creep Behaviour of Unreinforced PBT and Short-Fibre Reinforced PBT Using Prony and Fractional Derivative Models},
JOURNAL = {Applied Mechanics},
VOLUME = {6},
YEAR = {2025},
NUMBER = {3},
ARTICLE-NUMBER = {60},
URL = {https://www.mdpi.com/2673-3161/6/3/60},
ISSN = {2673-3161},
ABSTRACT = {This study presents an approach based on data-driven methods for determining the parameters needed to model time-dependent material behaviour. The time-dependent behaviour of the thermoplastic polymer polybutylene terephthalate is investigated. The material was examined under two conditions, one with and one without the inclusion of reinforcing short fibres. Two modelling approaches are proposed to represent the time-dependent response. The first approach is the generalised Maxwell model formulated through the classical exponential Prony series, and the second approach is a model based on fractional calculus. In order to quantify the comparative capabilities of both models, experimental data from tensile creep tests on fibre-reinforced polybutylene terephthalate and unreinforced polybutylene terephthalate specimens are analysed. A central contribution of this work is the implementation of a machine-learning-ready parameter identification framework that enables the automated extraction of model parameters directly from time-series data. This framework enables the robust fitting of the Prony-based model, which requires multiple characteristic times and stiffness parameters, as well as the fractional model, which achieves high accuracy with significantly fewer parameters. The fractional model benefits from a novel neural solver for fractional differential equations, which not only reduces computational complexity but also permits the interpretation of the fractional order and stiffness coefficient in terms of physical creep resistance. The methodological framework is validated through a comparative assessment of predictive performance, parameter cheapness, and interpretability of each model, thereby providing a comprehensive understanding of their applicability to long-term material behaviour modelling in polymer-based composite materials.},
DOI = {10.3390/applmech6030060}
}
```
---

