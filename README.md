# Stokes equations for non-Newtonian fluids with Input Convex Neural Networks (ICNN)

## Overview

This repository is dedicated to the study and analysis of fluid dynamics using Input Convex Neural Networks (ICNN) in comparison with traditional viscosity models. It features implementations for ICNN, scripts for model training and evaluation, convergence analysis, and real data examples.

Link for the scientific paper [currently under review]: [https://arxiv.org/abs/2401.07121](https://arxiv.org/abs/2401.07121)

## Repository Contents

### Scripts

1. **`convergence_script.py`**:
   - **Purpose**: Evaluates the convergence of fluid dynamics simulations using ICNN.
   - **Details**: This script sets up and runs fluid dynamic simulations with varying grid sizes to assess the convergence behavior of the ICNN model. It involves setting up the simulation environment, loading the ICNN model, and comparing the model's output with exact solutions.
   - **Key Components**: OpenMP settings, Firedrake for PDE solving, ICNN loading and evaluation, error computation, and convergence plotting.

2. **`ICNN.py`**:
   - **Purpose**: Defines the ICNN model used for approximating viscosity in fluid dynamics problems.
   - **Features**: The script includes the ICNN class with methods for creating the neural network, forward pass, data scaling, and model training with convex constraint. It supports ELU and ReLU activation functions.

3. **`realdata_example_plus_comparison.py`**:
   - **Purpose**: Demonstrates the application of the ICNN model using real-world data and compares its performance with traditional viscosity models.
   - **Functionality**: Includes functions for the Power Law and Carreau model, loads real flow data, trains the ICNN model, and plots comparisons between the ICNN predictions and traditional models.

### Data Files

1. **`flow_data.csv`**: Contains fluid data regarding the viscosity of aqueous solutions of Xanthan gum with sodium chloride addition. This data is used for training and evaluating the ICNN model on a real-world application.

2. **`ICNN_carreau_r16.pkl`** and **`ICNN_carreau_r28.pkl`**: Pre-trained ICNN models based on the Carreau law for different rheology parameters (r=1.6 and r=2.8). These models were generated from a sampled Carreau law with the respective exponent "r" and are serialized ICNN objects.

### Images

1. **`ICNN_models_comparison.png`**: An image produced by running `realdata_example_plus_comparison.py`, showcasing a comparison of the ICNN model's performance against traditional viscous laws. The plot highlights the superior performance of the ICNN model.

### Usage

#### Running the Scripts

1. **Convergence Analysis (`convergence_script.py`):**
   - To perform convergence analysis using the ICNN models, first set the path to the desired pre-trained ICNN model (either `ICNN_carreau_r16.pkl` or `ICNN_carreau_r28.pkl`) in the `icnn_file` variable within the `convergence_script.py`.
   - Run the script to evaluate the performance of the chosen ICNN model. The script will automatically load the model, run simulations on different grid sizes, and plot the convergence behavior.

2. **Model Training and Comparison (`realdata_example_plus_comparison.py`):**
   - This script uses the `flow_data.csv` file, which should be placed in the same directory as the script or its path updated in the script.
   - Running this script will train the ICNN model with the data from `flow_data.csv` and compare its performance against traditional viscosity models like the Power Law and Carreau models.
   - The script outputs graphical comparisons and performance metrics, such as R-squared and MSE, for each model.
