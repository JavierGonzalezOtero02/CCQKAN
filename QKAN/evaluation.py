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
from IPython.display import display
import re
import ast
import warnings

# My custom libraries
from QKAN import QKAN


########################## EVALUATION RELATED FUNCTIONS ##########################

def plot_actual_vs_predicted(Xs, Ys, Y_preds, errors):
    """
    Visualizes the best regression result (based on lowest error) using a 3D scatter plot of true vs. predicted values.

    Parameters:
    -----------
    Xs : list of pandas.DataFrame
        A list of training DataFrames, one per model or experiment.
        Each DataFrame must contain columns 'x0' and 'x1' used as input features.

    Ys : list of pandas.DataFrame
        A list of test DataFrames containing the ground truth target values.
        Each DataFrame must contain a column 'y'.

    Y_preds : list of array-like
        A list of arrays or lists, each containing predicted values from the model
        corresponding to each test dataset.

    errors : list of list of single-list floats e.g. [[[error1], [error2], ...], ...]
        A nested list containing error values (e.g., MSE) per prediction. 
        It is used to determine which model's result had the lowest error.

    Behavior:
    ---------
    - Selects the prediction with the lowest error across all attempts.
    - Plots a 3D scatter of:
        - True target values (`y`) in blue.
        - Predicted values (`y_pred`) in red.
    - Draws vertical lines connecting each true-predicted pair.
    - Displays interactive tooltips with feature and value information.
    - Adds a title showing the error of the selected model.

    Returns:
    --------
    None
        Displays an interactive Plotly 3D figure.
    """
    min_err = 10000
    index = 0
    for i, error in enumerate(errors):
        if error[0] < min_err:
            index = i
            min_err = error[0]
        
    X = np.array(Xs[index][['x0', 'x1']])
    Y = np.array(Ys[index]['y']).flatten()
    Y_pred = np.array(Y_preds[index]).flatten()

    # Scatter plot for true values
    true_scatter = go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=Y,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.6),
        name='True Y',
        text=[f'x1: {x1:.2f}, x2: {x2:.2f}, Y: {y:.2f}' for x1, x2, y in zip(X[:, 0], X[:, 1], Y)],
        hoverinfo='text'
    )

    # Scatter plot for predicted values
    pred_scatter = go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=Y_pred,
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.6),
        name='Predicted Y',
        text=[f'x1: {x1:.2f}, x2: {x2:.2f}, Y_pred: {yp:.2f}' for x1, x2, yp in zip(X[:, 0], X[:, 1], Y_pred)],
        hoverinfo='text'
    )

    # Lines connecting each true-predicted pair
    connection_lines = [
        go.Scatter3d(
            x=[X[i, 0], X[i, 0]],
            y=[X[i, 1], X[i, 1]],
            z=[Y[i], Y_pred[i]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False,
            hoverinfo='skip'
        )
        for i in range(len(X))
    ]

    # Create and display the figure
    fig = go.Figure(data=[true_scatter, pred_scatter] + connection_lines)
    fig.update_layout(
        title=f'3D Scatter Plot: True vs Predicted (MSE = {min_err:.4f})',
        scene=dict(
            xaxis_title='x1',
            yaxis_title='x2',
            zaxis_title='y',
        ),
        legend=dict(x=0.8, y=0.9)
    )
    fig.show()


def plot_training_loss_curves(global_train_error_evolution, ylabel):
    """
    Plots the training loss curves for multiple training attempts.

    Parameters:
    -----------
    global_train_error_evolution : list of list of float
        A list where each element is a list of loss values over training steps
        for a specific training attempt (e.g., different folds or initializations).

    Behavior:
    ---------
    - Plots one loss curve per training attempt.
    - Displays axis labels, a two-column legend, a grid, and adjusts the layout.
    - Limits the y-axis range to [0, 8] for better visualization.
    """
    plt.figure(figsize=(10, 6))  # Set figure size

    # Plot each training attempt
    for i, loss_curve in enumerate(global_train_error_evolution):
        plt.plot(loss_curve, label=f"Loss curve of attempt {i}")

    # Axis labels
    plt.xlabel("Training step", fontsize=12)
    plt.ylabel(ylabel, fontsize=12)

    # Y-axis limit
    #plt.ylim(0, 8)

    # Custom legend positioning
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)

    # Grid and layout
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def summarize_test_errors(global_test_errors):
    """
    Computes and returns summary statistics for a list of precomputed global test errors.

    Parameters:
    -----------
    global_test_errors : list or array-like
        A list of values representing the sum of absolute distances for each test point.

    Returns:
    --------
    pd.DataFrame
        A one-row DataFrame summarizing:
        - mean
        - median
        - min
        - max
        - std
    """
    errors = np.array(global_test_errors)

    stats = {
        ''
        'mean': np.mean(errors),
        'median': np.median(errors),
        'min': np.min(errors),
        'max': np.max(errors),
        'std': np.std(errors)
    }
    df = pd.DataFrame({
        'Statistic': list(stats.keys()),
        'Sum of Absolute Distances': list(stats.values())
    })

    return df.style.hide(axis='index')

def plot_test_error_distribution(local_test_errors, log_errors=True):
    """
    Visualizes the distribution of log10-transformed test errors across multiple training runs
    using vertical boxplots per test point (showing outliers).

    Parameters:
    -----------
    local_test_errors : list or np.ndarray of shape (n_runs, n_test_points)
        A 2D array where each row corresponds to a single training run, and each column to a test point.
        Example: 10 runs x 50 test points → shape (10, 50)

    log_errors : bool
        Whether to apply log10 transformation to the errors.

    Behavior:
    ---------
    - Applies log10 to errors (if requested).
    - Computes and plots mean, median, and interquartile range (Q1–Q3).
    - Displays boxplots per test point showing distribution across runs (with outliers).
    """
    errors = np.array(local_test_errors)
    errors = np.array(local_test_errors).reshape(errors.shape[0], errors.shape[1])


    # Safely apply log10
    epsilon = 1e-10
    errors = np.clip(errors, epsilon, None)
    if log_errors:
        errors = np.log10(errors)

    # Compute stats
    q1_vals = np.percentile(errors, 25, axis=0).ravel()
    q3_vals = np.percentile(errors, 75, axis=0).ravel()
    mean_vals = np.mean(errors, axis=0).ravel()
    median_vals = np.median(errors, axis=0).ravel()
    x = np.arange(1, errors.shape[1] + 1).ravel()


    plt.figure(figsize=(14, 6))

    # Shaded interquartile region
    plt.fill_between(x, q1_vals, q3_vals, color="skyblue", alpha=0.5, label="Q1–Q3 range")

    # Plot mean and median
    plt.plot(x, mean_vals, color="black", label="Mean", linewidth=2)
    plt.plot(x, median_vals, color="red", label="Median", linewidth=2)

    # Create boxplots
    box_data = [errors[:, i] for i in range(errors.shape[1])]
    plt.boxplot(
        box_data,
        positions=x,
        widths=0.3,
        patch_artist=False,
        showbox=False,               
        showcaps=True,
        showfliers=True,
        whiskerprops=dict(color="gray", linewidth=3),
        capprops=dict(color="gray", linewidth=3),
        medianprops=dict(color='none', linewidth=0),  
        flierprops=dict(markerfacecolor='blue', marker='o', markersize=4, linestyle='none', alpha=0.6)
    )

    # Labels
    plt.xlabel("Test Point Index")
    plt.ylabel("log₁₀(Loss)" if log_errors else "Absolute distances loss")
    plt.title("Boxplot of " + ("log₁₀ " if log_errors else "") + "Absolute Distance Errors Across Test Points")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()




def metrics_classification_EVQKAN(dataframes, model, parameters, plot=False):
    """
    Evaluates classification performance of a QKAN model across multiple datasets, 
    showing confusion matrices and key metrics (accuracy, precision, recall).

    Parameters:
    -----------
    dataframes : list of pandas.DataFrame
        List of test datasets, one per training attempt.
        Each DataFrame must contain 'x0', 'x1', and 'label' columns.
    
    model : QKAN
        The QKAN model used to generate predictions.

    parameters : list
        List of parameter sets, one per model for each dataset.

    plot : bool, optional (default: False)
        Whether to show a grouped bar chart comparing accuracy, precision, and recall across attempts.

    Returns:
    --------
    dict
        A dictionary containing:
        - 'accuracies': list of float
        - 'precisions': list of float
        - 'recalls': list of float
        - 'figures': list of matplotlib.figure.Figure (confusion matrix grid + optional bar chart)
    """

    accuracies = []
    precisions = []
    recalls = []
    figures = []

    n_attempts = len(dataframes)

    # Create a 2x5 grid for confusion matrices
    fig_grid, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()

    for i in range(n_attempts):
        # Prepare model input
        test_X0 = np.array(dataframes[i][['x0']].values.tolist(), dtype=float)
        y_pred = model.forward(test_X0, *parameters[i])
        test_X1 = np.array(dataframes[i][['x1']].values.tolist(), dtype=float)

        # Generate predicted labels: -1 if y_pred >= x1 else 1
        pred_labels = np.array([-1 if y.item() >= x[0] else 1 for y, x in zip(y_pred, test_X1)])
        test_labels = dataframes[i]['label'].to_numpy()

        # Compute classification metrics
        accuracy = np.mean(test_labels == pred_labels.flatten())
        precision = precision_score(test_labels, pred_labels, pos_label=1, zero_division=0)
        recall = recall_score(test_labels, pred_labels, pos_label=1, zero_division=0)

        accuracies.append(accuracy)
        precisions.append(precision)
        recalls.append(recall)

        # Display confusion matrix in the grid
        cm = confusion_matrix(test_labels, pred_labels, labels=[-1, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['-1', '1'])
        disp.plot(ax=axes[i], cmap='Blues', colorbar=False)
        axes[i].set_title(f'Attempt {i+1}', fontsize=12)
        axes[i].grid(False)

        # Add metrics below each matrix (clearly visible)
        axes[i].text(0.5, -0.25,
                     f'Acc: {accuracy:.2f}\nPrec: {precision:.2f}\nRec: {recall:.2f}',
                     transform=axes[i].transAxes,
                     ha='center', va='top',
                     fontsize=11)

    # Hide unused subplot cells if fewer than 10 attempts
    for j in range(len(axes)):
        if j >= n_attempts:
            axes[j].axis('off')

    # Final layout adjustments
    fig_grid.suptitle('Confusion Matrices with Metrics', fontsize=16)
    fig_grid.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    figures.append(fig_grid)

    # Optional grouped bar plot of metrics
    if plot:
        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        x_vals = np.arange(1, n_attempts + 1)
        width = 0.25

        mean_acc = np.mean(accuracies)
        mean_prec = np.mean(precisions)
        mean_rec = np.mean(recalls)
        
        ax_bar.bar(x_vals - width, accuracies, width=width, label=f"Accuracy (mean={mean_acc:.2f})", color="skyblue", edgecolor="black")
        ax_bar.bar(x_vals, precisions, width=width, label=f"Precision (mean={mean_prec:.2f})", color="lightgreen", edgecolor="black")
        ax_bar.bar(x_vals + width, recalls, width=width, label=f"Recall (mean={mean_rec:.2f})", color="salmon", edgecolor="black")


        ax_bar.set_xlabel('Attempt')
        ax_bar.set_ylabel('Score')
        ax_bar.set_title('Classification Metrics per Attempt')
        ax_bar.set_xticks(x_vals)
        ax_bar.set_xticklabels([str(i) for i in x_vals])
        ax_bar.set_ylim(0, 1)
        ax_bar.legend()
        ax_bar.grid(True)
        plt.tight_layout()
        plt.show()
        figures.append(fig_bar)

    return {
        "accuracies": accuracies,
        "precisions": precisions,
        "recalls": recalls,
        "figures": figures
    }


def plot_training_time_boxplots(time_data_per_task, model_names, task_name, show_stats=True):
    """
    Plots boxplots of training times for each model in a given task,
    and optionally displays a summary statistics table.

    Parameters
    ----------
    time_data_per_task : list of lists
        A list where each element is a list of training times (floats) 
        for a specific model within the task.
        
    model_names : list of str
        List of model names in the same order as the elements in `time_data_per_task`.
        
    task_name : str
        Name of the task (used as the plot title).

    show_stats : bool, default=True
        If True, displays a DataFrame with summary statistics (mean, median, std, min, max).

    Returns
    -------
    None
        Displays:
            - A boxplot per model
            - A DataFrame with statistics if show_stats=True
    """
    
    if len(time_data_per_task) != len(model_names):
        raise ValueError("The length of time_data_per_task must match the length of model_names.")
    
    df_list = []
    for model, times in zip(model_names, time_data_per_task):
        for t in times:
            df_list.append({"Model": model, "Training Time": t})
    
    df = pd.DataFrame(df_list)

    # Plot boxplot
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    
    ax = sns.boxplot(
        x="Model", 
        y="Training Time", 
        data=df, 
        showmeans=True,
        meanprops={"marker": "o", "markerfacecolor": "red", "markeredgecolor": "black"},
        boxprops=dict(alpha=0.7)
    )
    
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Mean',
                              markerfacecolor='red', markeredgecolor='black', markersize=8)]
    ax.legend(handles=legend_elements, loc='upper left')

    plt.title(f"Training Time per Attempt and Model - {task_name}", fontsize=14)
    plt.ylabel("Time (s)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Optionally show statistics
    if show_stats:
        summary_stats = df.groupby("Model")["Training Time"].agg(
            Mean="mean",
            Median="median",
            Std="std",
            Min="min",
            Max="max"
        ).round(2)

        display(summary_stats)


def summarize_total_training_time(time_data_per_model, model_names):
    """
    Calculates and displays the total training time for each model,
    preserving the original order of the model_names list.

    Parameters
    ----------
    time_data_per_model : list of lists
        A list where each element is a list of training times (floats) 
        for a specific model (e.g., from 10 attempts).
        
    model_names : list of str
        List of model names in the same order as the elements in `time_data_per_model`.

    Returns
    -------
    dict
        Dictionary where keys are model names and values are total training times (rounded to 2 decimals).
    """
    
    if len(time_data_per_model) != len(model_names):
        raise ValueError("The length of time_data_per_model must match the length of model_names.")
    
    total_times = {
        model: round(sum(times), 2)
        for model, times in zip(model_names, time_data_per_model)
    }

    # Convert to DataFrame for display
    df_total = pd.DataFrame({
        "Model": list(total_times.keys()),
        "Total Training Time (s)": list(total_times.values())
    })

    display(df_total.style.hide(axis='index'))
    return total_times


def plot_overall_training_heatmap(total_times_per_task_dict, average=False):
    """
    Plots a single heatmap to compare all models across all tasks.

    Parameters
    ----------
    total_times_per_task_dict : dict
        Dictionary where keys are task names, and values are dictionaries:
        { 'Task Name': { 'Model Name': total_time, ... }, ... }

    average : bool, default=False
        If True, divides total time by number of attempts (assumes 10), showing average time.

    Returns
    -------
    None
        Displays a heatmap with:
        - Rows as models
        - Columns as tasks
        - Values as total or average training time
    """

    # Build a DataFrame: tasks as columns, models as rows
    df = pd.DataFrame(total_times_per_task_dict).T  # tasks = rows, models = columns
    df = df.T  # Transpose: models = rows, tasks = columns

    if average:
        df = df / 10  # assuming 10 attempts

    plt.figure(figsize=(10, 6))
    sns.heatmap(df, annot=True, fmt=".1f", cmap="RdYlGn_r", linewidths=0.5, cbar_kws={"label": "Training Time (s)"})
    title = "Average Training Time per Model and Task" if average else "Total Training Time per Model and Task"
    plt.title(title)
    plt.xlabel("Task")
    plt.ylabel("Model")
    plt.tight_layout()
    plt.show()


def resources_from_config(config_str):
    """
    Parses a configuration string to extract architectural and trainable parameter details
    of a QKAN model. Builds the model and computes structural statistics.

    Parameters
    ----------
    config_str : str
        A string containing configuration data for model instantiation.
        Must include:
            - The network structure list (e.g., [1, 2, 1]) after 'optimizer,'
            - The integer max degree after the list
            - Boolean flags: GFCF, train_gfcf, train_angles

    Returns
    -------
    results : dict
        Dictionary containing:
            - 'n_gates' : int
                Total number of quantum gates used in the model circuit.
            - 'depth' : int
                Quantum circuit depth.
            - 'n_qubits' : int
                Number of qubits (wires) used in the model.
            - 'n_quantum_trainable_params' : int
                Number of quantum trainable parameters (e.g., QSVT angles if trainable).
            - 'n_classical_trainable_params' : int
                Number of classical trainable parameters (feature maps).
            - 'n_extra_classical_trainable_params' : int
                Additional classical parameters (e.g., eta, alpha, etc.).
            - 'n_trainable_params' : int
                Total number of trainable parameters (quantum + classical × 2).

    Notes
    -----
    This function instantiates a QKAN model with the parsed configuration.
    It suppresses warnings during model construction using `warnings.catch_warnings`.
    """
    n_gates = 0
    depth = 0
    n_qubits = 0
    n_quantum_trainable_params = 0 # Correspond to QSVT angles if trainable
    n_classical_trainable_params = 0
    n_extra_classical_trainable_params = 0
    n_trainable_params = 0
    
    # Extract network structure
    list_match = re.search(r'optimizer,\s*(\[[^\]]+\])', config_str)
    network_structure = ast.literal_eval(list_match.group(1)) if list_match else None

    # Extract max_degree
    num_match = re.search(r'optimizer,\s*\[[^\]]+\],\s*(\d+)', config_str)
    max_degree = int(num_match.group(1)) if num_match else None # degree_expansions attribute

    # Extract booleans
    def extract_bool(key):
        match = re.search(fr'{key}\s*=\s*(True|False)', config_str)
        return match.group(1) == "True" if match else None

    GFCF_par = extract_bool('GFCF')
    train_gfcf_par = extract_bool('train_gfcf')
    train_angles_par = extract_bool('train_angles')
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = QKAN(network_structure, max_degree, GFCF=GFCF_par, train_gfcf=train_gfcf_par, train_angles=train_angles_par)
    matrix = np.diag(np.array([0.5]*network_structure[0], requires_grad=False))
    dimension = 0
    n_gates = qml.specs(model._circuit)(matrix, 0)['resources'].num_gates
    depth  = qml.specs(model._circuit)(matrix, 0)['resources'].depth
    n_qubits = qml.specs(model._circuit)(matrix, 0)['resources'].num_wires
    n_quantum_trainable_params = qml.specs(model._circuit)(matrix, 0)['num_trainable_params']

    for i in range(model._number_layers):
        n_classical_trainable_params += model._N_list[i] * model._K_list[i] * (model._degree_expansions + 1)
    
    if (GFCF_par == True) and (train_gfcf_par == True): # Only GFCF 3 and 4 have 3 classical trainable parameters. All athers have 2.
        n_extra_classical_trainable_params = 3
    else:
        n_extra_classical_trainable_params = 2
    n_trainable_params = n_quantum_trainable_params + n_classical_trainable_params + n_classical_trainable_params

    results = {
        'n_gates': n_gates,
        'depth': depth,
        'n_qubits': n_qubits,
        'n_quantum_trainable_params': n_quantum_trainable_params,
        'n_classical_trainable_params': n_classical_trainable_params,
        'n_extra_classical_trainable_params': n_extra_classical_trainable_params,
        'n_trainable_params': n_trainable_params
    }
    return results


def resources_from_parameters(network_structure, max_degree, GFCF_par, train_gfcf_par, train_angles_par, return_key=None):
    """
    Builds a QKAN model with the specified configuration and computes its quantum and classical resource usage.

    Parameters
    ----------
    network_structure : list of int
        List defining the number of neurons in each layer of the quantum neural network (e.g., [1, 2, 1]).

    max_degree : int
        Maximum degree of polynomial expansion used in the classical feature maps.

    GFCF_par : bool
        Whether the model uses GFCF (Generalized Fourier Cosine Features).

    train_gfcf_par : bool
        Whether the GFCF parameters are trainable.

    train_angles_par : bool
        Whether the quantum QSVT angles are trainable.

    return_key : str or None, optional
        If provided, returns only the value associated with this key in the result dictionary.
        If None (default), returns the full dictionary.

    Returns
    -------
    dict or value
        Dictionary with resource metrics, or a single value if `return_key` is specified.
        
        Available keys:
            - 'n_gates'
            - 'depth'
            - 'n_qubits'
            - 'n_quantum_trainable_params'
            - 'n_classical_trainable_params'
            - 'n_extra_classical_trainable_params'
            - 'n_trainable_params'

    Notes
    -----
    - This function instantiates a QKAN model and computes resource statistics using `qml.specs`.
    - Warnings during model construction are suppressed.
    """

    n_gates = 0
    depth = 0
    n_qubits = 0
    n_quantum_trainable_params = 0
    n_classical_trainable_params = 0
    n_extra_classical_trainable_params = 0
    n_trainable_params = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = QKAN(network_structure, max_degree, GFCF=GFCF_par, train_gfcf=train_gfcf_par, train_angles=train_angles_par)

    matrix = np.diag(np.array([0.5]*network_structure[0], requires_grad=False))
    specs = qml.specs(model._circuit)(matrix, 0)
    n_gates = specs['resources'].num_gates
    depth = specs['resources'].depth
    n_qubits = specs['resources'].num_wires
    n_quantum_trainable_params = specs['num_trainable_params']

    for i in range(model._number_layers):
        n_classical_trainable_params += model._N_list[i] * model._K_list[i] * (model._degree_expansions + 1)

    n_extra_classical_trainable_params = 3 if GFCF_par and train_gfcf_par else 2
    n_trainable_params = n_quantum_trainable_params + n_classical_trainable_params + n_extra_classical_trainable_params

    results = {
        'n_gates': n_gates,
        'circuit depth': depth,
        'n_qubits': n_qubits,
        'n_quantum_trainable_params': n_quantum_trainable_params,
        'n_classical_trainable_params': n_classical_trainable_params,
        'n_extra_classical_trainable_params': n_extra_classical_trainable_params,
        'n_trainable_params': n_trainable_params
    }

    if return_key is not None:
        if return_key not in results:
            raise ValueError(f"'{return_key}' is not a valid key. Choose from: {list(results.keys())}")
        return results[return_key]

    return results



def plot_architecture_scaling(
    architectures,
    max_degrees,
    GFCF_par,
    train_gfcf_par,
    train_angles_par,
    x_metric='max_width',
    y_metric='n_gates'
):
    """
    Plots how a QKAN resource metric scales with different architectural properties.

    Parameters
    ----------
    architectures : list of list of int
        A list of network architectures (e.g., [[2,1], [3,1], [4,1], ...]).

    max_degrees : list of int
        A list of max_degree values, one per architecture.

    GFCF_par : bool
        Whether the model uses GFCF (Generalized Fourier Cosine Features).

    train_gfcf_par : bool
        Whether the GFCF parameters are trainable.

    train_angles_par : bool
        Whether the quantum QSVT angles are trainable.

    x_metric : str
        One of:
            - 'max_width': maximum width of each architecture
            - 'architecture_depth': number of layers in the architecture
            - 'max_degree': degree used in the classical feature maps

    y_metric : str
        One of the keys from `resources_from_parameters` to plot on the y-axis:
            - 'depth'
            - 'n_qubits'
            - 'n_gates'
            - 'n_quantum_trainable_params'
            - 'n_classical_trainable_params'
            - 'n_extra_classical_trainable_params'
            - 'n_trainable_params'

    Returns
    -------
    None
        Displays a matplotlib plot.
    """

    valid_y_metrics = [
        'circuit depth',
        'n_qubits',
        'n_gates',
        'n_quantum_trainable_params',
        'n_classical_trainable_params',
        'n_extra_classical_trainable_params',
        'n_trainable_params'
    ]

    valid_x_metrics = [
        'max_width',
        'architecture_depth',
        'max_degree'
    ]

    if y_metric not in valid_y_metrics:
        raise ValueError(f"'{y_metric}' is not a valid y_metric. Choose from: {valid_y_metrics}")
    if x_metric not in valid_x_metrics:
        raise ValueError(f"'{x_metric}' is not a valid x_metric. Choose from: {valid_x_metrics}")

    # Compute x-axis values
    if x_metric == 'max_width':
        x_vals = [max(arch) for arch in architectures]
    elif x_metric == 'architecture_depth':
        x_vals = [len(arch) for arch in architectures]
    else:  # 'max_degree'
        x_vals = max_degrees

    # Compute y-axis values
    y_vals = [
        resources_from_parameters(
            network_structure=arch,
            max_degree=deg,
            GFCF_par=GFCF_par,
            train_gfcf_par=train_gfcf_par,
            train_angles_par=train_angles_par,
            return_key=y_metric
        )
        for arch, deg in zip(architectures, max_degrees)
    ]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, marker='o')
    plt.xlabel(x_metric.replace('_', ' ').title())
    plt.ylabel(y_metric.replace('_', ' ').title())
    plt.title(f'{y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()