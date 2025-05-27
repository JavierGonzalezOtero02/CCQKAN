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

#My custom libraries
from QKAN import QKAN, optimization_evaluation as optim

def multiple_forwards(dataframes, model, parameters):
    """
    Evaluates classification performance of a QKAN classification model that simulates EVQKAN results
    over one or multiple datasets.

    Parameters:
    -----------
    dataframes : pandas.DataFrame or list of pandas.DataFrame
        One or several test datasets.
    model : QKAN
        The QKAN model used for predictions.
    parameters : list
        List of parameter sets, one for each model corresponding to the datasets.

    Returns:
    --------
    list of np.array
        Each entry is the predicted output for the corresponding dataset.
    """

    # Ensure dataframes is a list
    if isinstance(dataframes, pd.DataFrame):
        dataframes = [dataframes]
    
    y_preds = []

    for i in range(len(dataframes)):
        # Select input columns (exclude 'y' and 'label')
        input_columns = [col for col in dataframes[i].columns if col not in ['y', 'label']]
        test_X = np.array(dataframes[i][input_columns].values.tolist(), requires_grad=False)

        # Predict output using the model
        y_pred = model.Forward(test_X, *parameters[i])
        y_preds.append(np.array(y_pred).flatten())

    return y_preds

