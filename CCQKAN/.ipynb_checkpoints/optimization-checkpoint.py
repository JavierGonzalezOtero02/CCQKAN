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
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
import tracemalloc


########################## OPTIMIZATION RELATED FUNCTIONS ##########################
def MSE_error(y_pred, y):
    """
    Computes the Mean Squared Error (MSE) between predicted and true values.

    Parameters:
    -----------
    y_pred : array-like
        Matrix of predicted values (shape: [n_samples, n_labels]).
    y : array-like
        Matrix of true target values (same shape as y_pred).

    Returns:
    --------
    float
        The Mean Squared Error between predictions and targets.

    Raises:
    -------
    ValueError
        If the shapes of y_pred and y do not match.
    """
    y_pred = np.array(y_pred, requires_grad=False)
    y = np.array(y, requires_grad=False)

    if y_pred.shape != y.shape:
        raise ValueError("Shapes of y_pred and y must match.")
    squared_errors = (y_pred - y) ** 2
    mse = np.mean(squared_errors)
    
    return mse

def abs_error_EVQKAN_evaluation(y_pred, y):
    """
    Computes absolute error for QKAN predictions.

    Parameters:
    -----------
    y_pred : array-like
        Predicted outputs (1D list or array).
    y : array-like
        True output values (1D list or array).

    Returns:
    --------
    tuple
        A tuple (total_error, sample_errors), where:
        - total_error is the sum of weighted absolute errors.
        - sample_errors is a list of individual weighted errors.
    """
    error = 0
    errors = []
    for sample in range(len(y)):
        errors.append(abs(y_pred[sample] - y[sample]))
        error += errors[-1]
    return error, errors


def cost_qkan(model, X, Y, error_function, *parameters):
    """
    Computes the cost (MSE) of a QKAN model given input data and parameters.

    Parameters:
    -----------
    model : QKAN
        The QKAN model instance.
    X : array-like
        Input feature matrix of shape (n_samples, n_features).
    Y : array-like
        Ground truth labels of shape (n_samples, n_outputs).
    error_function: Callable
        Defined python function to calculate an error between predictions and real values
    *parameters : list of arrays
        Trainable parameters for each layer of the QKAN model.

    Returns:
    --------
    float
        Mean Squared Error of the model on the given input and targets.
    """
    #print('\n\n\nCOSTQKAN')
    #print(*parameters)
    #current, peak = tracemalloc.get_traced_memory()
    #print(f"Uso actual antes del forward: {current / 10**6:.2f} MB; pico: {peak / 10**6:.2f} MB")
    Y_pred = model.forward(X, *parameters)
    #current, peak = tracemalloc.get_traced_memory()
    #print(f"Uso actual despues del forward: {current / 10**6:.2f} MB; pico: {peak / 10**6:.2f} MB")
    #print(error_fn)
    #print(Y_pred)
    #print(Y)
    error = error_function(Y_pred, Y)
    return error