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
def plot_actual_vs_predicted(Xs, Ys, Y_preds, errors, error_metric='SAD', renderer='notebook'):
    import numpy as np
    import plotly.graph_objects as go

    # Find best prediction (lowest error)
    min_err = float('inf')
    index = 0
    for i, error in enumerate(errors):
        if error[0] < min_err:
            index = i
            min_err = error[0]

    X = np.array(Xs[index][['x0', 'x1']])
    Y = np.array(Ys[index]['y']).flatten()
    Y_pred = np.array(Y_preds[index]).flatten()

    # Legend-only markers
    legend_true = go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='blue'),
        name='True Y'
    )
    legend_pred = go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(size=10, color='red'),
        name='Predicted Y'
    )
    legend_error = go.Scatter3d(
        x=[None], y=[None], z=[None],
        mode='markers',
        marker=dict(color='rgba(0,0,0,0)'),
        name=f"{error_metric}: {min_err:.2f}"
    )

    # Actual points (not shown in legend)
    true_scatter = go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=Y,
        mode='markers',
        marker=dict(size=5, color='blue', opacity=0.6),
        showlegend=False,
        hoverinfo='text',
        text=[f'x0: {x0:.2f}, x1: {x1:.2f}, Y: {y:.2f}' for x0, x1, y in zip(X[:, 0], X[:, 1], Y)]
    )

    pred_scatter = go.Scatter3d(
        x=X[:, 0], y=X[:, 1], z=Y_pred,
        mode='markers',
        marker=dict(size=5, color='red', opacity=0.6),
        showlegend=False,
        hoverinfo='text',
        text=[f'x0: {x0:.2f}, x1: {x1:.2f}, Y_pred: {yp:.2f}' for x0, x1, yp in zip(X[:, 0], X[:, 1], Y_pred)]
    )

    connection_lines = [
        go.Scatter3d(
            x=[X[i, 0], X[i, 0]],
            y=[X[i, 1], X[i, 1]],
            z=[Y[i], Y_pred[i]],
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        )
        for i in range(len(X))
    ]

    fig = go.Figure(data=[
        legend_true, legend_pred, legend_error,
        true_scatter, pred_scatter, *connection_lines
    ])

    if renderer == 'notebook':
        fig.update_layout(
            width=250, height=250,
            scene=dict(
                xaxis_title='x0',
                yaxis_title='x1',
                zaxis_title='y',
            ),
            legend=dict(x=0.25, y=0.9, font=dict(size=8)),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig.show()
    else:
        fig.update_layout(
            width=1000, height=1000,
            scene=dict(
                xaxis_title='x0',
                yaxis_title='x1',
                zaxis_title='y',
            ),
            legend=dict(x=0.25, y=0.6, font=dict(size=35)),
            margin=dict(l=0, r=0, t=0, b=0)
        )
        fig.show(renderer='browser')


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
        pred_labels = np.array([-1 if y.item() >= np.sqrt(x[0]) else 1 for y, x in zip(y_pred, test_X1)])
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

def summary_classification_metrics(
    classification_metrics_0,
    classification_metrics_1,
    classification_metrics_2,
    classification_metrics_3,
    classification_metrics_4,
    classification_metrics_5,
    model_names,
    title="Classification Metrics Summary",
    output_path="classification_metrics_plot.pdf"
):
    all_metrics = [
        classification_metrics_0,
        classification_metrics_1,
        classification_metrics_2,
        classification_metrics_3,
        classification_metrics_4,
        classification_metrics_5
    ]

    metric_types = ["Accuracy", "Precision", "Recall"]
    metric_keys = {"Accuracy": "accuracies", "Precision": "precisions", "Recall": "recalls"}
    metric_colors = {
        "Accuracy": "#FA8072",   # salmon
        "Precision": "#90EE90",  # light green
        "Recall": "#ADD8E6"      # light blue
    }

    def to_float(x):
        try:
            return x.item()
        except AttributeError:
            return float(x)

    all_data = {metric: [] for metric in metric_types}
    means = {metric: [] for metric in metric_types}

    for model_metrics in all_metrics:
        for metric in metric_types:
            raw_values = model_metrics[metric_keys[metric]]
            floats = [to_float(v) for v in raw_values]
            all_data[metric].append(floats)
            means[metric].append(np.mean(floats))

    num_models = len(model_names)
    width = 0.25
    x_ticks = np.arange(num_models)

    fig, ax = plt.subplots(figsize=(16, 6))

    for x in x_ticks[:-1]:
        sep = (x + x + 1) / 2
        ax.axvline(x=sep, linestyle='--', color='gray', linewidth=0.6, alpha=0.6, zorder=0)

    for i, metric in enumerate(metric_types):
        color = metric_colors[metric]
        positions = x_ticks + (i - 1) * width
        box_data = all_data[metric]

        ax.boxplot(
            box_data,
            positions=positions,
            widths=width * 0.85,
            patch_artist=True,
            boxprops=dict(facecolor=color, edgecolor='black', linewidth=1.2),
            medianprops=dict(color='black', linewidth=2),
            whiskerprops=dict(color='black', linewidth=1.2),
            capprops=dict(color='black', linewidth=1.2),
            flierprops=dict(marker='o', markersize=4, linestyle='none',
                            markerfacecolor=color, markeredgecolor='black', alpha=0.6)
        )

        for j, mean_val in enumerate(means[metric]):
            pos = positions[j]
            ax.hlines(mean_val, pos - width * 0.4, pos + width * 0.4,
                      colors='black', linestyles='--', linewidth=1.5)

    ax.grid(True, axis='y', linestyle='--', linewidth=0.7)
    ax.grid(False, axis='x')  # quitar líneas verticales automáticas

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(model_names, rotation=0, ha='center', fontsize=20)
    ax.set_ylabel("Score", fontsize=23)
    ax.set_xlabel("")  
    ax.set_title(title, fontsize=23, pad=20)
    ax.tick_params(axis='y', labelsize=16)

    ax.set_ylim(0, 1)

    handles = [
        plt.Line2D([0], [0], color=metric_colors["Accuracy"], lw=10),
        plt.Line2D([0], [0], color=metric_colors["Precision"], lw=10),
        plt.Line2D([0], [0], color=metric_colors["Recall"], lw=10),
        plt.Line2D([0], [0], color='black', lw=2, label='Median'),
        plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='Mean'),
    ]
    labels = ["Accuracy", "Precision", "Recall", "Median", "Mean"]
    ax.legend(handles, labels, fontsize=16, loc='lower left', frameon=False)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.2)

    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.show()  
    plt.close()


def plot_error_summary(
    error_lists,
    model_names,
    task,
    y_label="Test Error",
    title="Test Error Summary",
    output_path="test_error_summary.pdf"
):
    """
    Plots a boxplot summary of test error distributions for multiple models, including
    mean and median lines, and annotated statistics per model.

    Parameters:
    -----------
    error_lists : list of list of float or tensor
        A list containing sublists of test error values for each model.
        Each sublist corresponds to one model and must contain numerical values
        (or PyTorch-style tensors with `.item()` method).
    
    model_names : list of str
        Names of the models to be used as x-axis labels. The order must match `error_lists`.

    task : str
        Task for which the plot is being created

    y_label : str, optional (default="Test Error")
        Label to display on the Y-axis.

    title : str, optional (default="Test Error Summary")
        Title to display at the top of the plot.

    output_path : str, optional (default="test_error_summary.pdf")
        File path to save the resulting plot as a PDF.

    Returns:
    --------
    None
        The function saves the plot to the specified PDF file and displays it interactively.

    Features:
    ---------
    - Boxplots with pastel blue color per model.
    - Median as a solid black line, mean as a dashed black line.
    - Annotated text boxes for each model showing:
        * Mean (μ), Median (Md), Standard deviation (σ), Min, Max.
    - Text box positions are customized per model to avoid overlaps.
    - Clean layout with axis styling and no x-axis label.

    Example:
    --------
    >>> plot_error_summary(
    ...     error_lists=[[1.2, 1.3, 1.1], [0.9, 1.0, 1.1]],
    ...     model_names=["Model A", "Model B"],
    ...     y_label="MAE",
    ...     title="Model Comparison",
    ...     output_path="mae_comparison.pdf"
    ... )
    """
    def to_float(x):
        try:
            return x.item()
        except AttributeError:
            return float(x)

    # Convert all values to float (e.g., from PyTorch tensors)
    data = [[to_float(val) for val in model_errors] for model_errors in error_lists]

    fig, ax = plt.subplots(figsize=(16, 6))

    # Vertical separator lines between models
    for x in range(len(model_names)):
        sep = (x + x + 1) / 2
        ax.axvline(x=sep, linestyle='--', color='gray', linewidth=0.6, alpha=0.6, zorder=0)

    # Draw the boxplots
    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        boxprops=dict(facecolor="#ADD8E6", edgecolor='black', linewidth=1.2),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, linestyle='none',
                        markerfacecolor="#ADD8E6", markeredgecolor='black', alpha=0.6)
    )

    # Add mean lines and statistical annotations
    for i, values in enumerate(data):
        mean = np.mean(values)
        ax.hlines(mean, i + 1 - 0.25, i + 1 + 0.25,
                  colors='black', linestyles='--', linewidth=1.5)

        stats_values = {
            'max': np.max(values),
            'Md': np.median(values),
            'μ': mean,
            'min': np.min(values),
            'σ': np.std(values)
        }
        
        stats_str = '\n'.join(f"{k}={v:.2f}" for k, v in stats_values.items())

        if task == 'classification':
            # Custom position for each label block
            model = model_names[i]
            y_pos = None
            va = 'bottom'
    
            if model == "Plain-CHEB":
                y_pos = min(values) - (max(values) - min(values)) * 0.3
                va = 'top'
            elif model == "Plain-Flex":
                y_pos = max(values) - 3.4
            elif model == "GFCF-CHEB-0":
                y_pos = min(values) - (max(values) - min(values)) * 0.3
                va = 'top'
            else:
                y_pos = max(values) + 0.5 + (max(values) - min(values)) * 0.15
        elif task == 'exponential':
            # Custom position for each label block
            model = model_names[i]
            y_pos = None
            va = 'bottom'
    
            if model == "Plain-CHEB":
                y_pos = min(values) - (max(values) - min(values)) * 0.3
                va = 'top'
            elif model == "Plain-Flex":
                y_pos = min(values) - 16
            elif model == "GFCF-CHEB-0":
                y_pos = min(values) - 3
                va = 'top'
            else:
                y_pos = max(values) + 3
        elif task == 'polynomial':
            # Custom position for each label block
            model = model_names[i]
            y_pos = None
            va = 'bottom'
    
            if model == "Plain-CHEB":
                y_pos = min(values) - 1.45
                va = 'top'
            elif model == "Plain-Flex":
                y_pos = max(values) + 1.45
            elif model == "GFCF-CHEB-0":
                y_pos = min(values) - 1.45
                va = 'top'
            else:
                y_pos = max(values) + 1.45
        

        ax.text(
            i + 1,
            y_pos,
            stats_str,
            fontsize=15,
            fontweight='bold',
            ha='center',
            va=va,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', alpha=0.9)
        )

    # Axis styling
    ax.set_xticks(np.arange(1, len(model_names) + 1))
    ax.set_xticklabels(model_names, fontsize=20, rotation=0)
    ax.set_ylabel(y_label, fontsize=23)
    ax.set_xlabel("")  # Remove "Model" label
    ax.set_title(title, fontsize=23, pad=20)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.7)
    ax.set_xlim(0.5, len(model_names) + 0.5)

    # Legend
    handles = [
        plt.Line2D([0], [0], color='black', lw=2, label='Median'),
        plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='Mean'),
    ]
    labels = ["Median", "Mean"]
    ax.legend(handles, labels, fontsize=16, loc='lower left', frameon=False)

    # Save and display
    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.2)
    fig.savefig(output_path, format='pdf', bbox_inches='tight')  # Save to PDF
    plt.show()  # Display the figure
    plt.close()


def plot_training_time_boxplots(
    time_data_per_task,
    model_names,
    title="Training Time Summary",
    y_label="Training Time (s)",
    output_path="training_time_summary.pdf",
    show_stats=True
):
    """
    Plots a boxplot summary of training times for multiple models, including mean as red dot
    and vertical separators between model groups.

    Parameters
    ----------
    time_data_per_task : list of list of float
        Training times per model.

    model_names : list of str
        Model labels in the same order.

    title : str
        Title shown at the top of the plot.

    y_label : str
        Label for the Y-axis.

    output_path : str
        Path to save the figure.

    show_stats : bool
        If True, print summary statistics table.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd

    def to_float(x):
        try:
            return x.item()
        except AttributeError:
            return float(x)

    data = [[to_float(t) for t in times] for times in time_data_per_task]
    x_ticks = np.arange(len(model_names))

    fig, ax = plt.subplots(figsize=(16, 6))

    for x in range(len(model_names)):
        sep = (x + x + 1) / 2
        ax.axvline(x=sep, linestyle='--', color='gray', linewidth=0.6, alpha=0.6, zorder=0)

    box = ax.boxplot(
        data,
        patch_artist=True,
        widths=0.6,
        boxprops=dict(facecolor="#ADD8E6", edgecolor='black', linewidth=1.2),
        medianprops=dict(color='black', linewidth=2),
        whiskerprops=dict(color='black', linewidth=1.2),
        capprops=dict(color='black', linewidth=1.2),
        flierprops=dict(marker='o', markersize=4, linestyle='none',
                        markerfacecolor="#ADD8E6", markeredgecolor='black', alpha=0.6)
    )

    for i, values in enumerate(data):
        mean = np.mean(values)
        ax.plot(i + 1, mean, 'o', color='red', markeredgecolor='black', markersize=6, zorder=3)

    ax.set_xticks(np.arange(1, len(model_names) + 1))
    ax.set_xticklabels(model_names, fontsize=20, rotation=0)
    ax.set_ylabel(y_label, fontsize=23)
    ax.set_xlabel("")
    ax.set_title(title, fontsize=23, pad=20)
    ax.tick_params(axis='y', labelsize=16)
    ax.grid(True, axis='y', linestyle='--', linewidth=0.7)
    ax.set_xlim(0.5, len(model_names) + 0.5)

    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color='black', lw=2, label='Median'),
        Line2D([0], [0], marker='o', color='w', label='Mean',
               markerfacecolor='red', markeredgecolor='black', markersize=8)
    ]
    ax.legend(handles=handles, fontsize=16, loc='upper left', frameon=False)

    plt.subplots_adjust(left=0.06, right=0.98, top=0.90, bottom=0.2)
    fig.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    if show_stats:
        df_list = []
        for model, times in zip(model_names, data):
            for t in times:
                df_list.append({"Model": model, "Training Time": t})
        df = pd.DataFrame(df_list)
        summary_stats = df.groupby("Model")["Training Time"].agg(
            Mean="mean", Median="median", Std="std", Min="min", Max="max"
        ).round(2)

        from IPython.display import display
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


import warnings

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
def plot_architecture_scaling(architectures, max_degrees, GFCF_par, train_gfcf_par, train_angles_par,
                               x_metric='max_width', y_metric='n_gates', output_path='architecture_scaling.pdf'):
    """
    Plots how a QKAN resource metric scales with architectural properties and saves the plot as a PDF.

    Parameters
    ----------
    architectures : list of list of int
        List of network architectures (e.g., [[2,1], [3,1], ...]).

    max_degrees : list of int
        List of max_degree values for each architecture.

    GFCF_par : bool
        Whether GFCF is used.

    train_gfcf_par : bool
        Whether GFCF parameters are trainable.

    train_angles_par : bool
        Whether QSVT angles are trainable.

    x_metric : str, default='max_width'
        Metric for x-axis: 'max_width', 'architecture_depth', or 'max_degree'.

    y_metric : str, default='n_gates'
        Metric for y-axis. One of:
        ['circuit depth', 'n_qubits', 'n_gates', 'n_quantum_trainable_params',
         'n_classical_trainable_params', 'n_extra_classical_trainable_params', 'n_trainable_params'].

    output_path : str, default='architecture_scaling.pdf'
        Path to save the resulting figure as a PDF.

    Returns
    -------
    None
        Displays and saves the figure.
    """
    import matplotlib.pyplot as plt

    valid_y_metrics = [
        'circuit depth', 'n_qubits', 'n_gates', 'n_quantum_trainable_params',
        'n_classical_trainable_params', 'n_extra_classical_trainable_params', 'n_trainable_params'
    ]
    valid_x_metrics = ['max_width', 'architecture_depth', 'max_degree']

    if y_metric not in valid_y_metrics:
        raise ValueError(f"'{y_metric}' is not a valid y_metric. Choose from: {valid_y_metrics}")
    if x_metric not in valid_x_metrics:
        raise ValueError(f"'{x_metric}' is not a valid x_metric. Choose from: {valid_x_metrics}")

    # Compute X values
    if x_metric == 'max_width':
        x_vals = [max(arch) for arch in architectures]
    elif x_metric == 'architecture_depth':
        x_vals = [len(arch) for arch in architectures]
    else:
        x_vals = max_degrees

    # Compute Y values
    y_vals = [
        resources_from_parameters(
            network_structure=arch,
            max_degree=deg,
            GFCF_par=GFCF_par,
            train_gfcf_par=train_gfcf_par,
            train_angles_par=train_angles_par,
            return_key=y_metric
        ) for arch, deg in zip(architectures, max_degrees)
    ]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, linewidth=2)
    plt.xlabel(x_metric.replace('_', ' ').title(), fontsize=14)
    plt.ylabel(y_metric.replace('_', ' ').title(), fontsize=14)
    plt.title(f'{y_metric.replace("_", " ").title()} vs {x_metric.replace("_", " ").title()}', fontsize=15)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    plt.show()
