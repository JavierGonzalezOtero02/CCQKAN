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
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import sys
import pickle

########################## DATASET RELATED FUNCTIONS ##########################


#------------------------ Dataset Generators ------------------------#

def generate_dataset_regression(n_values, domains, function, n_dimensions, seed):
    """
    Generates a synthetic dataset for regression tasks.

    Parameters:
    -----------
    n_values : int
        Number of data samples (rows) to generate.
    domains : list of tuple
        List of (min, max) intervals specifying the sampling range for each ionput dimension.
        The length of this list must match `n_dimensions`.
    function : callable
        A target function to approximate. It should accept a 1D array of length `n_dimensions`
        as ionput and return a scalar output.
    n_dimensions : int
        Number of ionput variables (features) for each sample.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame with `n_dimensions` ionput columns named 'x0', 'x1', ..., 'x(n-1)',
        and one output column named 'y', containing the result of applying `function` to each row.

    Example:
    --------
    >>> def f(x): return x[0]**2 + onp.sin(x[1])
    >>> df = generate_dataset_regression(1000, [-1, 1], f, 2)
    """
    onp.random.seed(seed)  # For reproducibility

    # Generate n_dimensions of random ionputs in domain
    assert len(domains) == n_dimensions, "Must be one domain per dimension."

    # Generamos los datos respetando el dominio por dimensión
    X = onp.array([
        onp.random.uniform(low=domains[dim][0], high=domains[dim][1], size=n_values)
        for dim in range(n_dimensions)
    ]).T

    # Compute the output of the function for each row of ionputs
    y = onp.apply_along_axis(function, 1, X)

    # Create column names x1, x2, ..., xn
    col_names = [f'x{i}' for i in range(n_dimensions)]

    # Create the DataFrame with ionput columns and output
    df = pd.DataFrame(X, columns=col_names)
    df['y'] = y

    return df

def generate_classification_dataset(n_values, domains, function, n_dimensions, seed):
    """
    Generates a synthetic dataset for binary classification based on a non-linear decision boundary.

    Parameters:
    -----------
    n_values : int
        Number of data samples to generate.
    domains : list of tuple
        List of (min, max) intervals specifying the sampling range for each input dimension.
        The length of this list must match `n_dimensions`.
    function : callable
        A function defining the decision boundary. It should accept a 1D input array
        of length `n_dimensions` and return a scalar value.
    n_dimensions : int
        Number of input dimensions (features) for the dataset.

    Returns:
    --------
    pandas.DataFrame
        A DataFrame containing:
        - One input column per feature ('x0', 'x1', ..., 'xn').
        - A 'label' column with values -1 or 1, based on whether x1 <= f(x).
        - A 'y' column with the normalized computed function value used to assign the label.

    Notes:
    ------
    The classification rule assigns:
    - Label -1 if x1 <= f(x)
    - Label  1 otherwise

    The function sets a random seed (0) for reproducibility.
    """
    onp.random.seed(seed)  # For reproducibility

    # Verify that there is one domain per dimension
    assert len(domains) == n_dimensions, "Must be one domain per dimension."

    # Generate input data respecting each dimension's domain
    X = onp.array([
        onp.random.uniform(low=domains[dim][0], high=domains[dim][1], size=n_values)
        for dim in range(n_dimensions)
    ]).T  # shape (n_values, n_dimensions)

    # Evaluate the function
    y = onp.apply_along_axis(function, 1, X)

    # Normalize y between 0 and 1
    y_min = y.min()
    y_max = y.max()
    y_normalized = (y - y_min) / (y_max - y_min)

    # Assign labels based on the normalized function
    labels = onp.where(np.sqrt(X[:, 1]) <= y_normalized, -1, 1)

    # Create the DataFrame
    col_names = [f'x{i}' for i in range(n_dimensions)]
    df = pd.DataFrame(X, columns=col_names)
    df['label'] = labels
    df['y'] = y_normalized

    return df



#------------------------ Plot Datasets ------------------------#


def plot_classification_unidimensional(df, highlight_index=None):
    """
    Plots a 2D binary classification dataset using the 'label' column for coloring.
    Optionally highlights a single point.

    Parameters:
    -----------
    df : pandas.DataFrame
        Must contain columns 'x0', 'x1', and 'label'.
    
    highlight_index : int, optional
        If provided, only the point at this index will be shown (highlighted),
        while the full classification space is still displayed in the background.
    """
    plt.figure(figsize=(8, 6))

    if highlight_index is not None:
        # Background points (faded)
        colors = df['label'].map({-1: 'red', 1: 'blue'})
        plt.scatter(df['x0'], df['x1'], c=colors, alpha=0.1)

        # Highlighted point
        point = df.iloc[highlight_index]
        color = 'red' if point['label'] == -1 else 'blue'
        plt.scatter(point['x0'], point['x1'], c=color, edgecolor='black', s=100, label='Highlighted Point')
    else:
        # Normal full scatter plot
        colors = df['label'].map({-1: 'red', 1: 'blue'})
        plt.scatter(df['x0'], df['x1'], c=colors, alpha=0.7)

    # Custom legend
    red_patch = plt.Line2D([0], [0], marker='o', color='w', label='Label -1',
                            markerfacecolor='red', markersize=8)
    blue_patch = plt.Line2D([0], [0], marker='o', color='w', label='Label 1',
                             markerfacecolor='blue', markersize=8)

    handles = [red_patch, blue_patch]
    if highlight_index is not None:
        highlight_dot = plt.Line2D([0], [0], marker='o', color='w', label='Highlighted Point',
                                   markerfacecolor=color, markeredgecolor='black', markersize=10)
        handles.append(highlight_dot)

    plt.legend(handles=handles)

    # Styling
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.title('1D Classification (From DataFrame)')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def plot_regression_dataset(dataset, n_dimensions, title="Dataset Plot"):
    """
    Plots an interactive 2D or 3D scatter plot of a regression dataset using Plotly.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        A DataFrame containing input feature columns ('x1', 'x2', ...) and a target column 'y'.
        For 1D input, 'x1' and 'y' must be present.
        For 2D input, 'x1', 'x2', and 'y' must be present.

    n_dimensions : int
        Dimensionality of the input features. Must be either 1 or 2.
        - 1: creates a 2D scatter plot (x1 vs y).
        - 2: creates a 3D scatter plot (x1, x2, y).

    title : str, optional
        Title of the plot (default is "Dataset Plot").

    Raises:
    -------
    ValueError
        If `n_dimensions` is not 1 or 2.

    Behavior:
    ---------
    - For 1D inputs, it creates a simple 2D scatter plot.
    - For 2D inputs, it creates an interactive 3D scatter plot.
    - The figure is displayed automatically.
    """
    if n_dimensions == 1:
        fig = px.scatter(
            dataset,
            x='x0',
            y='y',
            title=title,
            labels={'x0': 'x0', 'y': 'y'},
            opacity=0.8
        )
        fig.update_traces(marker=dict(size=6))

    elif n_dimensions == 2:
        fig = px.scatter_3d(
            dataset,
            x='x0',
            y='x1',
            z='y',
            title=title,
            labels={'x0': 'x0', 'x1': 'x1', 'y': 'y'},
            opacity=0.8
        )
        fig.update_traces(marker=dict(size=4))

    else:
        raise ValueError("Only 1D or 2D input can be visualized (n_dimensions must be 1 or 2).")

    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    fig.show()


#------------------------ Functions to generate synthetic datasets ------------------------#

def unidimensional_hyperplane(x):
    """
    Computes a non-linear function of x[0] using a reproducible, locally seeded random parameter vector.

    Parameters:
    -----------
    x : array-like
        Input vector (only x[0] is used).

    Returns:
    --------
    float
        Non-linear function value at x[0] based on fixed pseudo-random parameters.
    """
    rng = onp.random.default_rng(seed=1)  # Local, fixed seed generator
    d = rng.uniform(0, 1, size=8)

    return (onp.exp(d[0] * np.sqrt(x[0]) + d[1])
            + d[2] * onp.sqrt(1 - d[3] * np.sqrt(x[0])**2)
            + onp.cos(d[4] * np.sqrt(x[0]) + d[5])
            + onp.sin(d[6] * np.sqrt(x[0]) + d[7]))



def multidimensional_exponential(x):
    """
    Computes a non-linear exponential function over a 4D ionput.

    Parameters:
    -----------
    x : array-like
        Ionput vector of length 4.

    Returns:
    --------
    float
        Result of exp(sin(x0² + x1²) + sin(x2² + x3²)).
    """
    return onp.exp(onp.sin(x[0]**2 + x[1]**2) + onp.sin(x[2]**2 + x[3]**2))

def multidimensional_polynomial(x):
    """
    Computes a simple normalized polynomial function over a 2D ionput.

    Parameters:
    -----------
    x : array-like
        Ionput vector of length 2.

    Returns:
    --------
    float
        The value of (x0² + x1²)
    """
    return (x[0]**2 + x[1]**2)