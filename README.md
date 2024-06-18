# RNN and GRU Language Modeling

This repository contains an implementation for a project involving Recurrent Neural Networks (RNNs) and Gated Recurrent Units (GRUs) for language modeling. 

## Overview

1. **Training RNNs and GRUs:**
    * Implementing the predict method in `rnn.py` for forward prediction in an RNN.
    * Implementing `compute_loss` and `compute_mean_loss` methods in `runner.py` to compute total and average losses.
    * Implementing `acc_deltas` for truncated backpropagation and `acc_deltas_bptt` for backpropagation through time (BPTT).
    * Implementing similar methods in `gru.py` for GRU-specific training and predictions.

2. **Language Modeling:**
    * Performing parameter tuning using different combinations of learning rates, hidden units, and steps for truncated backpropagation.
    * Training an RNN and a GRU on a larger dataset using the best parameter settings found from tuning and evaluating the models' performance.

3. **Predicting Agreement with RNNs and GRUs:**
    * Testing whether an RNN and a GRU can learn agreement rules in English by training the models to predict whether a verb is singular or plural based on the preceding words.

The repository includes the following files:

* **rnn.py:** Implementation of RNN methods.
* **gru.py:** Implementation of GRU methods.
* **runner.py:** Training models and defining loss functions.
* **rnnmath.py:** Helper functions for matrix and vector operations.
* **requirements.txt:** Specification file for creating a Python virtual environment with necessary packages.


## Results

### Training RNNs (Task 1)

We trained RNN models with varying parameters to evaluate their performance in language modeling. Key findings include:

- Experimented with different combinations of hidden layers, lookback steps, and learning rates.
- Observed that increasing the number of hidden layers generally decreases mean loss, indicating improved model performance.
- Identified optimal parameters (**50 hidden units, 0 lookback, 0.5 learning rate**) that performed best on a larger training set.
- Graphed mean loss and learning rate over epochs, demonstrating convergence and model improvement.

### Language Modeling (Task 2)

Parameter tuning involved testing 18 combinations to optimize model performance:

| Combination | Hidden Layers | Lookback | Learning Rate | Mean Loss | Adjusted Mean Loss |
|-------------|---------------|----------|---------------|-----------|--------------------|
| 1           | 25            | 0        | 0.5           | 5.01697   | 5.39493            |
| ...         | ...           | ...      | ...           | ...       | ...                |
| 18          | 50            | 5        | 0.05          | 5.31422   | 5.73137            |

- Found that higher hidden layers generally improve performance, except in a few cases.
- Noted the impact of learning rate on model exploration and convergence speed.
- Achieved a mean loss of 4.42700 on the test set with optimal parameters.

### Predicting Agreement with RNNs and GRUs (Task 3)

Implemented and compared RNNs and GRUs for predicting agreement, focusing on:

- Updating weights using specified formulae.
- Forward prediction functions implemented based on equations provided.
- Compared accuracy between RNN and GRU models with varying parameters, highlighting the impact of hidden layers on accuracy improvement.
- Examined mean loss and accuracy graphs over epochs.

### Comparing Recurrent Models (Task 4)

Explored differences between RNNs and GRUs:

- Highlighted GRU's advantage in handling long-term dependencies using update and reset gates.
- Discussed the impact of BPTT on model performance and the vanishing gradient problem.
- Hypothesized and tested different scenarios, including the effect of lookback steps and training size on model accuracy.
- Presented results and graphs supporting hypotheses.

For detailed graphs, results, and code implementations, please refer to the appendices and source code in this repository.




