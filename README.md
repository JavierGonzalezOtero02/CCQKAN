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
> [Flexible Quantum Kolmogorov Arnold Networks](./Thesis report/Flexible_Quantum_Kolmogorov_Arnold_Networks.pdf)

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

## Features

- Supports:
  - Classical preprocessing with fixed or trainable **GFCF** (`alpha`)
  - Quantum architectures: **CHEB-QKAN** and **Flex-QKAN**
  - Complete training + evaluation on synthetic datasets
- Based on [PennyLane](https://pennylane.ai) for quantum simulation
- Modular and extendable design

---

## Reproducing Results

To run an experiment:

```bash
python main.py --model flex --task regression --alpha-trainable

