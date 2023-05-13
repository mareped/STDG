# STDG - Synthetic Tabular Data Generation

Code for my Master Thesis: "Exploring the Value of GANs for Synthetic Tabular Data Generation in Healthcare with a Focus on Data Quality, Augmentation, and Privacy".

This project utilizes Python 3.8.

## Repository Structure

This repository contains the code, data, and results related to my thesis on synthetic data generation in healthcare. Below is a brief description of each folder:

- `/data`: This directory includes both the original and synthetic data for all datasets used in the thesis. Data is stored in CSV format.

- `/evaluation`: Contains code related to the evaluation of synthetic datasets. Notably, it includes the implementation of the "SynthEval: Synthetic Data Classifier Evaluation" framework. Additionally, all code related to the SDMetrics library and TableEvaluator is located here.

- `/generators`: Here, you can find the implementation of the GAN models, namely CopulaGAN and CTGAN, utilized in the study. These models are implemented with the SDV library.

- `/results`: This directory includes all results related to the datasets used in the thesis. 

- `/saved_models`: In this folder, you will find the trained GAN models saved in .pkl format.

- `/utilities`: Contains additional code related to data pre-processing, plotting, and other supplementary functions used throughout the project.

Please navigate through these directories to explore the project further.

## Example: Running the SynthEval Framework
Code snippet from  `/evaluation/run_clf_framework.py` :
```python
# Declare which classifiers to use
logreg = LogisticRegression()
rf = RandomForestClassifier()
mlp = MLPClassifier()

evaluator = SynthEval(real_path, fake_path, result_path)

# Add all classifiers to the evaluator
evaluator.add_all_classifiers(logreg, rf, mlp)

# Compare datasets performance
evaluator.compare_datasets_performance(real_percentage=0.5, synth_percentage=1, cross_val=True) 

```
This script starts by declaring the classifiers to be used (Logistic Regression, Random Forest, and MLP Classifier), and initializes the SynthEval evaluator with the real data, synthetic data and result path. The classifiers are then added to the evaluator, and finally, the script compares the performance of the datasets.
