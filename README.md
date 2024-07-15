# ICNN-Stokes

This repository contains the code and data used to reproduce the results presented in the paper [Input-Convex Neural Networks for Stokes Flow Modeling](https://arxiv.org/abs/2401.07121).

## Directory Structure

Each folder in this repository corresponds to a specific figure or table in the paper, and contains the scripts and data required to reproduce the results.

- **Carreau (ICNN training, Table_6)**
  - Contains the training script for the Carreau rheological model.
  - **Files**:
    - `training_carreau.py`: Trains the ICNN on Carreau rheological data.
    - `parameter_ass_A_tuner.py`: Numerically determines the constants to verify Assumptions A.
    - Contains the script to reproduce Table 6.

- **Figure_1**
  - Scripts and data to reproduce Figure 1 in the paper.

- **Figure_2-3**
  - Scripts and data to reproduce Figures 2 and 3 in the paper.

- **Figure_4**
  - Scripts and data to reproduce Figure 4 in the paper.

- **Figure_5-6-7**
  - Scripts and data to reproduce Figures 5, 6, and 7 in the paper.

- **Figure_8**
  - Scripts and data to reproduce Figure 8 in the paper.

- **Figure_9**
  - Scripts and data to reproduce Figure 9 in the paper.
 
- **Figure_10**
  - Scripts and data to reproduce Figure 10 in the paper.

- **NaCL (ICNN training, Table_4)**
  - Contains the training script for real datasets with varying molar concentrations of NaCl.
  - **Files**:
    - `training_realdata.py`: Trains the ICNN on the "flow_data_XX" datasets, where XX can be 00, 01, 05, or 07. In the paper, these datasets are referred to as NaCL{XX}_XG.
      - **Dataset Description**: The solutions differ by molar concentration \( M \) of NaCl, ranging between 0 and 0.7, with a constant Xanthan gum content of 1 g/L for all solutions.
    - `parameter_ass_A_tuner.py`: Determines the constants for verifying Assumptions A.
    - Contains the scripts to reproduce Table 4.

- **Table_1**
  - Scripts and data to reproduce Table 1 in the paper.
 
- **Table_2-3**
  - Scripts and data to reproduce Table 2 and 3 in the paper.

- **Table_5**
  - Scripts and data to reproduce Table 5 in the paper.
 
- **Table_8**
  - Scripts and data to reproduce Table 8 in the paper.

## General Notes

- The `ICNN.py` file contains the definition of the Input-Convex Neural Network (ICNN). This file appears in multiple directories because the training specifications for the ICNN differ depending on the application.
- Communication between different scripts is facilitated using pickle variables.

## Requirements

Ensure you have the required packages installed before running the scripts. You can install the dependencies using:

```bash
pip install -r requirements.txt
```

*Important*: Firedrake requires a special installation process. To install Firedrake:

- Firedrake requires a Unix-like operating system (e.g., Linux or macOS).
- Ensure you have Python 3.7 or newer.
  
Open a terminal and run the following command to install Firedrake:

```bash
curl -O https://raw.githubusercontent.com/firedrakeproject/firedrake/master/scripts/install_firedrake.sh
bash install_firedrake.sh
```

This script will install Firedrake and its dependencies.
After installation, activate the Firedrake environment:

```bash
source firedrake/bin/activate
```

Once Firedrake is installed and the environment is activated, you can run your scripts as usual.

For more detailed information and troubleshooting, refer to the [Firedrake documentation](https://www.firedrakeproject.org/documentation.html).

