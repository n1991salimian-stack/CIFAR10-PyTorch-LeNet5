# LeNet-5 Image Classification (PyTorch)

This repository contains a clean implementation of the classic **LeNet-5** Convolutional Neural Network (CNN) trained on the **CIFAR-10** dataset using **PyTorch**. It demonstrates a fundamental deep learning pipeline, including data preprocessing, custom model architecture, and a training loop with validation.

The script trains the model for 20 epochs using the Adam optimizer. It tracks training and validation accuracy in real-time using a progress bar and concludes by generating a plot to visualize the model's performance and learning curve.

## Usage
Ensure you have `torch`, `torchvision`, `matplotlib`, and `tqdm` installed, then run:

```bash
python train_lenet.py
