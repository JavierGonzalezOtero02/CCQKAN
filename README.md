# CCQKAN: Classical-to-Classical Quantum Kolmogorov-Arnold Networks

This repository contains the complete implementation of **CCQKAN**, a framework for building, training, and evaluating **Quantum Kolmogorov-Arnold Networks (QKANs)** with classical inputs and outputs.

The repository accompanies the thesis submitted to the international conference **Quantum Techniques in Machine Learning (QTML)**.  

---

## Project Overview

This project introduces two main contributions to enhance the expressivity of QKANs:

- **GFCF Models**  
  Use of *Generalized Fractional-order Chebyshev Functions* as a classical processing step before quantum encoding.

- **Flex-QKAN**  
  A flexible QKAN architecture that introduces learnable basis functions using trainable angles via the *Quantum Singular Value Transformation (QSVT)* framework.

Both approaches are evaluated on classification and regression tasks using synthetically generated datasets.

Full methodology, background, and results are presented in the accompanying Bachelor's thesis:

> **Flexible Quantum Kolmogorov-Arnold Networks**  
> *Javier González Otero*  
> Supervised by *Miguel Ángel González Ballester*  
> Academic Year: 2024–2025  
> [Flexible Quantum Kolmogorov-Arnold Networks](./Thesis%20report/Flexible_Quantum_Kolmogorov_Arnold_Networks.pdf)

---

## Repository Structure

```text
├── CCQKAN/                    # Main framework: CCQKAN class, training & evaluation scripts
│
├── CCQKAN_datasets/          # Synthetic dataset generators and data visualization utilities
│
├── results/                  # Experimental results obtained for the three evaluation tasks presented in the thesis document
│
├── QKAN_multiple_trainings.py   # Script for running multiple training iterations (as in the thesis)
│
├── Quantum_Kolmogorov_Arnold_Networks_TFG.pdf  # Final thesis document (project report)
│
├── Results_notebook.ipynb    # Jupyter notebook showing performance metrics and comparisons
│
└── slides.pdf                # Slides used for the thesis presentation
```


---

## Models compatible with CCQKAN:

<p align="center">
  <img src="./Thesis%20report/images/Models_CCQKAN.png" width="600"/>
</p>

The following table shows the six model variants that can be constructed using the CCQKAN framework, along with their corresponding configuration arguments.
| Model          | GFCF flag | Trainable GFCF flag| Train angles flag |
|----------------|-----------|--------------------|-------------------|
| Plain-CHEB     | False     | -                  | False             |
| Plain-Flex     | False     | -                  | True              |
| GFCF-CHEB-0    | True      | False              | False             |
| GFCF-CHEB-1    | True      | True               | False             |
| GFCF-Flex-0    | True      | False              | True              |
| GFCF-Flex-1    | True      | True               | True              |

---
## How to construct and train CCQKAN models

To create one of the previous 6 CCQKAN models simply import the class and initialize it with the desired arguments:

```python
from CCQKAN import CCQKAN

model = CCQKAN(...)
```
Data has to be initialized as an N-dimensional list or array-like object. For example:

```python
matrix = [1, 2, 3, 4, 5]
```

Data can then be ingested by the model through the forward method. Due to PennyLane optimization procedure, parameters must be included as arguments in the forward pass:

```python
actual_parameters = [getattr(model, attr) for attr in dir(model) if attr.startswith('_parameters_')]
result = model.forward(matrix, *actual_parameters)
```

To train the model, define an optimizer compatible with the PennyLane framework and execute the following line where GFCF and train_gfcf are flags:
```python
trained_parameters, cost_values = train_loop_qkan(model, steps, matrix_x, matrix_target, optimizer, GFCF, train_gfcf, training_error_function=optim.MSE_error)
```
---
## Replicate the experiments shown in results notebook

To replicate the experiments of this project run the CCQKAN_multiple_trainings.py script using the following arguments:

```bash
python QKAN_multiple_trainings.py <mode> <seed> <number_of_runs> <output_filename>
```
Arguments:

<mode>: experiment type -> classification_unidimensional_hyperplane, regression_multidimensional_exponential or regression_multidimensional_polynomial

<seed>: random seed for reproducibility. This project has used seed 0 for regression experiments and seed 1 for classification experiments.

<number_of_runs>: number of independent training runs. 10 were conducted for the experiments issued in this work.

<output_filename>: name of the output .pkl file to store results (currently available ones are stored in the [results](./results))
