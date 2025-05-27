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

    return df

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


        