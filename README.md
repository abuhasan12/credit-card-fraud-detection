# Credit Card Fraud Detection

This project was undertaken to build a credit card fraud detection model.
The model was trained using a dataset for Credit Card Fraud Detection model found at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.
The dataset contains transactions made by credit cards in September 2013 by European cardholders.

## Data

The dataset represents a common problem of an imbalanced dataset.
It contains 284,807 transactions, of which 0.172% are fraudelent transactions.
The features of the dataset include time (which is the amount of seconds elapsed since the first transaction),
transaction amount, and 28 unknown features which are the result of PCA performed on the original dataset to protect the privacy of the credit card owners.

## Model

The chosen model is a **voting classifier** created using scikit-learn consisting of:
* A support-vector-machine classifier
* A KNN classifier
* A Bagged Logstic Regression classifier

The metrics used to evaluate the validation performance of the final classifier were recall and specificity:
* Recall - to assess the model's performance in classifying fraud transactions
* Specificity - to assess the model's performance in classifying non-fraud transactions

To read more about the problem investigation and development, please refer to the notebook provided.