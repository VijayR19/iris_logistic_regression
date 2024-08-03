# Iris Logistic Regression
This repository contains logistic regression analyses of the Iris dataset using both Python scripts and Jupyter Notebooks. The Iris dataset, a staple in machine learning, includes three classes of iris plants with four features: sepal length, sepal width, petal length, and petal width.

# Overview
## Jupyter Notebook (iris.ipynb)
Purpose: Provides an interactive environment for exploring data and performing logistic regression analysis with detailed explanations.
Features:
Data Visualization: Plots to illustrate data distribution and feature relationships.
Step-by-Step Analysis: Detailed breakdown of the logistic regression process.
Evaluation Metrics: Confusion matrix and metrics like precision, recall, and F1-score.
Benefits: Ideal for educational purposes and experimentation with interactive visuals.

## Python Script (main.py)
Purpose: A standalone program for efficient logistic regression analysis, suitable for automation.
Features:
Efficient Execution: Streamlined implementation for quick processing.
Reusability: Contains reusable functions for other projects.
Scalability: Easily integrates into batch processing workflows.
Benefits: Faster execution and better suited for production environments.
Accuracy and Model Evaluation
Model Training: Both implementations use logistic regression to classify iris plants into Setosa, Versicolour, and Virginica with an 80/20 train-test split.

# Jupyter Notebook:
Accuracy: Approximately 96% on the test set.
Evaluation: Confusion matrix and classification report for detailed performance analysis.

# Python Script:
Accuracy: Achieves perfect accuracy of 100% on the test set.
Evaluation: Outputs accuracy and confusion matrix efficiently.
Improvements
Feature Scaling: Standardization to enhance model performance.
Hyperparameter Tuning: Potential adjustments for improved accuracy.

## Files

- `iris.ipynb`: Jupyter Notebook for interactive data analysis and visualization.
- `main.py`: Python script for training and evaluating a logistic regression model.

## Installation

To run this project, you need the following Python packages:

- scikit-learn
- pandas
- matplotlib
- seaborn

Install them using pip:

```bash
pip install scikit-learn pandas matplotlib seaborn
