# CCQKAN: Classical-to-Classical Quantum Kolmogorov-Arnold Networks

This repository contains the complete implementation of **CCQKAN**, a framework for building, training, and evaluating **Quantum Kolmogorov-Arnold Networks (QKANs)** with classical inputs and outputs.

---

## 🧠 Project Overview

This project introduces two main contributions to enhance the expressivity of QKANs:

- **GFCF Models**  
  Use of *Generalized Fractional-order Chebyshev Functions* as a classical preprocessing step before quantum encoding.

- **Flex-QKAN**  
  A flexible QKAN architecture that introduces learnable basis functions using trainable angles via the *Quantum Singular Value Transformation (QSVT)* framework.

Both approaches are evaluated on classification and regression tasks using synthetically generated datasets.

📄 Full methodology, background, and results are presented in the accompanying Bachelor's thesis:

> **Flexible Quantum Kolmogorov-Arnold Networks**  
> *Javier González Otero*  
> Supervised by *Miguel Ángel González Ballester*  
> Academic Year: 2024–2025  
> [📄 Download PDF](./Quantum_Kolmogorov_Arnold_Networks_TFG.pdf)

---

## 🔗 Paper Reference

> This repository accompanies the thesis submitted to the international conference **Quantum Techniques in Machine Learning (QTML)**.  
>  
> 📌 Citation (BibTeX) coming soon.

---

## 📁 Repository Structure

```text
CCQKAN/
├── models/       # CCQKAN model definitions (CHEB-QKAN, Flex-QKAN, etc.)
├── circuits/     # Quantum circuit components (BE construction, QSVT, etc.)
├── data/         # Scripts for synthetic dataset generation
├── training/     # Training loop, losses, and utility functions
├── notebooks/    # Jupyter Notebooks with experiment visualizations
└── main.py       # Entry point for training models


---

## ⚙️ Features

- Supports:
  - Classical preprocessing with fixed or trainable **GFCF** (`alpha`)
  - Quantum architectures: **CHEB-QKAN** and **Flex-QKAN**
  - Complete training + evaluation on synthetic datasets
- Based on [PennyLane](https://pennylane.ai) for quantum simulation
- Modular and extendable design

---

## 🧪 Reproducing Results

To run an experiment:

```bash
python main.py --model flex --task regression --alpha-trainable

