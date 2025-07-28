# Lung Disease Image Classification using ResNet-18
Dataset: https://drive.google.com/file/d/1qMiBYwDkytBVsN0_K5iONw6trHPhbjI0/view?usp=sharing

This repository contains a PyTorch script for training a ResNet-18 model from scratch to classify lung cancer images. The model is trained on a dataset provided in `.npz` format and implements modern deep learning techniques to achieve high accuracy. The final model achieves **~90.0% accuracy** on the test set.

## Overview

The primary goal of this project is to build an effective image classifier for a multi-class lung cancer dataset. Instead of using a pre-trained model, this script demonstrates how to train a `ResNet-18` architecture from random initialization, leveraging strong data augmentation and regularization techniques to prevent overfitting and improve generalization.

## Key Features

- **Model**: `ResNet-18` trained from scratch.
- **Data Handling**: Loads image and label data from `.npz` files for training, validation, and testing.
- **Strong Data Augmentation**: Utilizes `TrivialAugmentWide` and `RandomErasing` to create a robust training set.
- **Advanced Training Techniques**:
    - **Label Smoothing**: Helps regularize the model and prevent overconfidence.
    - **Adam Optimizer**: An efficient and popular optimization algorithm.
    - **Cosine Annealing Scheduler**: Dynamically adjusts the learning rate during training for better convergence.
    - **Gradient Clipping**: Prevents exploding gradients, ensuring stable training.
    - **Early Stopping**: Monitors validation accuracy to stop training when performance plateaus, saving time and preventing overfitting.
- **Evaluation**:
    - Standard evaluation on the test set.
    - **Test-Time Augmentation (TTA)**: Averages predictions over original and horizontally-flipped images to boost final performance.

## Performance

The model was trained and evaluated, yielding the following results on the test set:

- **Final Test Accuracy (Standard)**: `92.16%`
- **Final Test Accuracy (with TTA)**: `92.14%`

The best performing model checkpoint is saved to `best_lung_cancer_model_resnet18_from_scratch.pth`.
