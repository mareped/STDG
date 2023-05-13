# STDG - Synthetic Tabular Data Generation

Code for my Master Thesis: "Exploring the Value of GANs for Synthetic Tabular Data Generation in Healthcare with a Focus on Data Quality, Augmentation, and Privacy"

## Repository Structure

This repository contains the code, data, and results related to my thesis on synthetic data generation in healthcare. Below is a brief description of each folder:

- **data**: This directory includes both the original and synthetic data for all datasets used in the thesis. Data is stored in CSV format.

- **evaluation**: Contains code related to the evaluation of synthetic datasets. Notably, it includes the implementation of the "SynthEval: Synthetic Data Classifier Evaluation" framework. Additionally, all code related to the SDMetrics library and TableEvaluator is located here.

- **generators**: Here, you can find the implementation of the GAN models, namely CopulaGAN and CTGAN, utilized in the study. These models are implemented with the SDV library.

- **results**: This directory includes all results related to the datasets used in the thesis. 

- **saved_models**: In this folder, you will find the trained GAN models saved in .pkl format.

- **utilities**: Contains additional code related to data pre-processing, plotting, and other supplementary functions used throughout the project.

Please navigate through these directories to explore the project further.

## Example of running the SynthEal Framework:
