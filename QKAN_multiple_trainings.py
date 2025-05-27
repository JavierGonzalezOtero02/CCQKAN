import sys
import pickle
import pennylane as qml
import gc, sys
import time

# My custom libraries
from QKAN import QKAN as qk, training as tr, optimization_evaluation as optim
from QKAN_datasets import datasets_utils as dt

########################## MAIN ##########################

if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise ValueError("Wrong arguments. Usage: python script.py <task> <pkl filename>\nExample: python script.py regression_multidimensional_exponential results/resulting_data.pkl")

    mode = sys.argv[1]
    seed = int(sys.argv[2])
    n_experiments = int(sys.argv[3])
    data_results_filename = sys.argv[4]

    train_data = []
    test_data = []
    parameters_pre_train = []
    parameters_post_train = []
    local_test_errors = []
    global_test_errors = []
    global_train_error_evolution = []
    test_preds = []
    max_mins = []
    configuration = ""    # Returnable variable to store configuration of the experiment
    time_per_attempt = []


    # TRAIN AND EVALUATE MULTIPLE TIMES

    if mode == "regression_multidimensional_exponential":
        print(f'{n_experiments} will be conducted')

        configuration = f"seed: {seed}, n_experiments: {n_experiments}, test: dt.generate_dataset_regression(50, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)], dt.multidimensional_exponential, 4, seed), train: dt.generate_dataset_regression(50, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)], dt.multidimensional_exponential, 4, seed+1+i), optimizer: qml.AdamOptimizer(stepsize=0.3), args_train: train_df, test_df, ['x0', 'x1', 'x2', 'x3'], optimizer, [4,1], 2, [-1,1], 55, [train_min_y, train_max_y], GFCF=True, eta=1, alpha=0.8, train_gfcf=True, train_angles=True, time: 26_05_19_17"
        
        test_df = dt.generate_dataset_regression(50, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)], dt.multidimensional_exponential, 4, seed)
        for i in range(n_experiments):
            gc.collect()
            print('Experiment: ', i)
            train_df = dt.generate_dataset_regression(50, [(-1, 1), (-1, 1), (-1, 1), (-1, 1)], dt.multidimensional_exponential, 4, seed+1+i)
            train_max_y = train_df['y'].max()
            train_min_y = train_df['y'].min()
            max_mins.append([train_min_y, train_max_y])
            optimizer = qml.AdamOptimizer(stepsize=0.3)
            train_data_iter, test_data_iter, parameters_pre_train_iter, parameters_post_train_iter, local_test_errors_iter, global_test_errors_iter, global_train_error_evolution_iter, test_preds_iter, attempt_training_time = tr.training_evaluate_multiple_times(train_df, test_df, ['x0', 'x1', 'x2', 'x3'], optimizer, [4,1], 2, [-1,1], 55, [train_min_y, train_max_y], GFCF=True, eta=1, alpha=0.8, train_gfcf=True, train_angles=True)
            train_data = train_data + train_data_iter
            test_data = test_data + test_data_iter
            parameters_pre_train = parameters_pre_train + parameters_pre_train_iter
            parameters_post_train = parameters_post_train + parameters_post_train_iter
            local_test_errors = local_test_errors + local_test_errors_iter
            global_test_errors = global_test_errors + global_test_errors_iter
            global_train_error_evolution = global_train_error_evolution + global_train_error_evolution_iter
            test_preds = test_preds + test_preds_iter
            time_per_attempt.append(attempt_training_time)

    elif mode == "classification_unidimensional_hyperplane":
        print(f'{n_experiments} will be conducted')

        configuration = f"rng_seed: 1, seed: {seed}, n_experiments: {n_experiments}, test: dt.generate_classification_dataset(50, [(0, 1), (0, 1)], dt.unidimensional_hyperplane, 2, seed), train: dt.generate_classification_dataset(45, [(0, 1), (0, 1)], dt.unidimensional_hyperplane, 2, seed+1+i), optimizer: qml.AdamOptimizer(stepsize=0.3), args_train: train_df, test_df, ['x0'], optimizer, [1, 2, 1], 2, [0,1], 20, [train_min_y, train_max_y], GFCF=True, eta=1, alpha=0.8, train_gfcf=True, train_angles=True, time: 26_05_19_24"
        
        test_df = dt.generate_classification_dataset(50, [(0, 1), (0, 1)], dt.unidimensional_hyperplane, 2, seed)
        for i in range(n_experiments):
            gc.collect()
            print('Experiment: ', i)
            train_df = dt.generate_classification_dataset(45, [(0, 1), (0, 1)], dt.unidimensional_hyperplane, 2, seed+1+i)
            train_max_y = train_df['y'].max()
            train_min_y = train_df['y'].min()
            max_mins.append([train_min_y, train_max_y])
            optimizer = qml.AdamOptimizer(stepsize=0.3)
            train_data_iter, test_data_iter, parameters_pre_train_iter, parameters_post_train_iter, local_test_errors_iter, global_test_errors_iter, global_train_error_evolution_iter, test_preds_iter, attempt_training_time = tr.training_evaluate_multiple_times(train_df, test_df, ['x0'], optimizer, [1, 2, 1], 2, [0,1], 20, [train_min_y, train_max_y], GFCF=True, eta=1, alpha=0.8, train_gfcf=True, train_angles=True)
            train_data = train_data + train_data_iter
            test_data = test_data + test_data_iter
            parameters_pre_train = parameters_pre_train + parameters_pre_train_iter
            parameters_post_train = parameters_post_train + parameters_post_train_iter
            local_test_errors = local_test_errors + local_test_errors_iter
            global_test_errors = global_test_errors + global_test_errors_iter
            global_train_error_evolution = global_train_error_evolution + global_train_error_evolution_iter
            test_preds = test_preds + test_preds_iter
            time_per_attempt.append(attempt_training_time)

    elif mode == "regression_multidimensional_polynomial":
        print(f'{n_experiments} will be conducted')

        configuration = f"seed: {seed}, n_experiments: {n_experiments}, test: dt.generate_dataset_regression(50, [(-1, 1), (-1, 1)], dt.multidimensional_polynomial, 2, seed), train: dt.generate_dataset_regression(100, [(-1, 1), (-1, 1)], dt.multidimensional_polynomial, 2, seed+1+i), optimizer: qml.AdamOptimizer(stepsize=0.3), args_train: train_df, test_df,['x0', 'x1'], optimizer, [2,1], 2, [-1,1], 100, [train_min_y, train_max_y], GFCF=True, eta=1, alpha=0.8, train_gfcf=True, train_angles=True, time: 26_05_19_31"
        
        test_df = dt.generate_dataset_regression(50, [(-1, 1), (-1, 1)], dt.multidimensional_polynomial, 2, seed)
        for i in range(n_experiments):
            gc.collect()
            print('Experiment: ', i)
            train_df = dt.generate_dataset_regression(100, [(-1, 1), (-1, 1)], dt.multidimensional_polynomial, 2, seed+1+i)
            train_max_y = train_df['y'].max()
            train_min_y = train_df['y'].min()
            max_mins.append([train_min_y, train_max_y])
            optimizer = qml.AdamOptimizer(stepsize=0.3)
            train_data_iter, test_data_iter, parameters_pre_train_iter, parameters_post_train_iter, local_test_errors_iter, global_test_errors_iter, global_train_error_evolution_iter, test_preds_iter, attempt_training_time = tr.training_evaluate_multiple_times(train_df, test_df,['x0', 'x1'], optimizer, [2,1], 2, [-1,1], 100, [train_min_y, train_max_y], GFCF=True, eta=1, alpha=0.8, train_gfcf=True, train_angles=True)
            train_data = train_data + train_data_iter
            test_data = test_data + test_data_iter
            parameters_pre_train = parameters_pre_train + parameters_pre_train_iter
            parameters_post_train = parameters_post_train + parameters_post_train_iter
            local_test_errors = local_test_errors + local_test_errors_iter
            global_test_errors = global_test_errors + global_test_errors_iter
            global_train_error_evolution = global_train_error_evolution + global_train_error_evolution_iter
            test_preds = test_preds + test_preds_iter
            time_per_attempt.append(attempt_training_time)

    # SAVE RESULTS
    
    data_to_save = {
        "train_data": train_data,
        "test_data": test_data,
        "parameters_pre_train": parameters_pre_train,
        "parameters_post_train": parameters_post_train,
        "local_test_errors": local_test_errors,
        "global_test_errors": global_test_errors,
        "global_train_error_evolution": global_train_error_evolution,
        "test_preds": test_preds,
        "min_y_max_y": max_mins,
        "configuration": configuration,
        "time_per_attempt": time_per_attempt
    }
    
    with open(data_results_filename, "wb") as f:
        pickle.dump(data_to_save, f)
