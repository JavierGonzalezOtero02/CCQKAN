import numpy as onp
from numpy.polynomial.chebyshev import Chebyshev
import pennylane as qml
from pennylane.templates.state_preparations.mottonen import compute_theta, gray_code
from pennylane import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import torch
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import tracemalloc
import psutil, os
import gc, sys


# My custom libraries
from QKAN import QKAN, optimization as optim

########################## TRAINING RELATED FUNCTIONS ##########################
def train_loop_qkan(model, steps, X, Y, optimizer, GFCF, train_gfcf, training_error_function=optim.MSE_error):
    """
    Trains a QKAN model using a custom optimization loop overa fixed number of steps.

    Parameters:
    ----------- 
    model : QKAN
        The QKAN model to be trained.
    steps : int
        Number of training steps (iterations) to perform.
    X : np.ndarray
        Input feature matrix of shape (n_samples, n_features).
    Y : np.ndarray
        Target values of shape (n_samples, n_outputs).
    optimizer : Optimizer
        Optimizer object compatible with PennyLane, e.g., qml.AdamOptimizer.
    training_error_function : callable, optional
        Function used to compute the loss during training (default: MSE).
        Must accept (y_pred, y_true) and return a scalar loss.

    Returns:
    --------
    tuple
        - parameters : list of updated parameter arrays for each layer.
        - cost_vals : list of cost values (MSE) recorded at each training step.
    """
    # Dynamically extract all model attributes that represent layer parameters
    parameters = [getattr(model, attr) for attr in dir(model) if attr.startswith('_parameters_')]
    #print('type!!!')
    #print(type(parameters))
    cost_vals = []
    start_time = time.time()
    for i in range(steps):
        gc.collect()
        #process = psutil.Process(os.getpid())
        #print(f"RAM usada por el proceso step de train {i}: {process.memory_info().rss / 1024 ** 2:.2f} MB")
        
        # Define a cost function that takes a variable number of parameters
        def cost_func(*params):
            #print('COST_FUNC!')
            #print(*params)
            return optim.cost_qkan(model, X, Y, training_error_function, *params)
        
        # Perform one optimization step
        #current, peak = tracemalloc.get_traced_memory()
        #print(f"Uso actual antes de step_cost: {current / 10**6:.2f} MB; pico: {peak / 10**6:.2f} MB")
        values, cost = optimizer.step_and_cost(cost_func, *parameters)
        if GFCF and train_gfcf:
            if values[-2] <= 0:
                values[-2] = np.array([0.001], requires_grad=train_gfcf)
        parameters = values
        if 0 < i < steps - 1:
            print(f'Cost step {i-1}: {cost}')
            cost_vals.append(cost)
        elif i == steps - 1:
            print(f'Cost step {i-1}: {cost}')
            cost_vals.append(cost)
            last_cost = optim.cost_qkan(model, X, Y, training_error_function, *parameters)
            print(f'Cost step {i}: {last_cost}')
            cost_vals.append(last_cost)
    end_time = time.time()
    training_time = end_time - start_time
        
    return parameters, cost_vals, training_time

def training_evaluate_multiple_times(train_df, test_df, data_cols, optimizer, architecture, max_degree, range_values, train_iterations, range_values_output, GFCF=False, eta=1, alpha=1, train_gfcf=True, train_angles=True, training_error_function=optim.MSE_error, evaluation_error=optim.abs_error_EVQKAN_evaluation):
    """
    Repeats the training and evaluation of DIFFERENT QKAN models as in the EVQKAN paper.
    (Can also be used for a single training run by setting n_times=1.)

    Parameters:
    -----------
    df : pandas.DataFrame
        The full dataset containing input features and target labels.
    data_cols : list of str
        Names of the columns to use as input features.
    optimizer : Optimizer
        Optimizer object compatible with PennyLane, e.g., qml.AdamOptimizer.
    architecture : list of int
        Neural network structure defining the number of neurons per layer (e.g., [4, 1]).
    max_degree : int
        Maximum degree for the Chebyshev polynomial expansions in each layer.
    range_values : list or tuple
        Output value range for the QKAN model (e.g., [-1, 1]).
    train_iterations : int
        Number of training steps per trial.
    range_values_output : 2-D list or tuple of float [min, max]
        Initial reconstruction values.
    n_times : int
        Number of cross-validation trials (splits) to perform.
    training_error_function : callable, optional
        Function used to compute the loss during training (default: MSE).
        Must accept (y_pred, y_true) and return a scalar loss.
    evaluation_error : callable, optional
        Function used to evaluate model performance on the test set (default: abs_error_EVQKAN_evaluation).
        Should return both the global and local (per-sample) error.

    Returns:
    --------
    tuple
        A collection of lists, one for each trial:
        - train_data : list of DataFrames for training sets.
        - test_data : list of DataFrames for testing sets.
        - parameters_pre_train : model parameters before training (per trial).
        - parameters_post_train : trained model parameters (per trial).
        - local_test_errors : per-sample test errors (weighted absolute error).
        - global_test_errors : total test errors per trial.
        - global_train_error_evolution : MSE over iterations for each trial.
    """
    # 10 TRAINING TRIALS WITH DIFFERENT FOLDS OF TRAINING TEST DATA WILL BE PERFORMED
    
    # Variables to store
    train_data = []
    test_data = []
    
    parameters_pre_train = []
    parameters_post_train = []
    
    local_test_errors = []
    global_test_errors = []
    
    global_train_error_evolution = []

    test_preds = []
    
    gc.collect()
    #process = psutil.Process(os.getpid())
    #print(f"RAM usada por el proceso en iteración {i}: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    

    # Create train and test data for concrete split
    train_data.append(train_df)
    test_data.append(test_df)

    # Create variables as matrix to pass to functions
    train_X = np.array(train_df[data_cols].values.tolist(), requires_grad=False)
    test_X = np.array(test_df[data_cols].values.tolist(), requires_grad=False)
    
    train_Y = np.array(train_df[['y']].values.tolist(), requires_grad=False)
    test_Y = np.array(test_df[['y']].values.tolist(), requires_grad=False)

    # Create a new model for each trial
    #current, peak = tracemalloc.get_traced_memory()
    #print(f"Uso actual antes de crear modelo en training multipletimes: {current / 10**6:.2f} MB; pico: {peak / 10**6:.2f} MB")
    qkan = QKAN(architecture, max_degree, GFCF=GFCF, eta=eta, alpha=alpha, train_gfcf=train_gfcf, train_angles=train_angles, range_values=range_values, range_values_output=range_values_output)
    parameters_pre_train.append([getattr(qkan, attr) for attr in dir(qkan) if attr.startswith('_parameters_')]) # To use them in forward -> *parameters_pre_train[i]

    # Train it
    parameters, cost_vals, attempt_training_time = train_loop_qkan(qkan, train_iterations, train_X, train_Y, optimizer, GFCF, train_gfcf, training_error_function=training_error_function) # parameters are in list format hence * is required
    parameters_post_train.append(parameters)
    global_train_error_evolution.append(cost_vals)

    # Test it
    Y_pred_test = qkan.forward(test_X, *parameters)
    test_preds.append(np.array(Y_pred_test).flatten())
    global_error, point_errors = evaluation_error(Y_pred_test, test_Y)
    print('TEST ERROR')
    print(global_error)
    local_test_errors.append(point_errors)
    global_test_errors.append(global_error)
    return train_data, test_data, parameters_pre_train, parameters_post_train, local_test_errors, global_test_errors, global_train_error_evolution, test_preds, attempt_training_time