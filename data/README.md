# Credit Card Fraud Detection Dataset

This dataset was retrieved from Kaggle. It contains credit card transactions made in September 2013 by European cardholders.

## Data

The dataset represents a common problem of an imbalanced dataset.
It contains 284,807 transactions, of which 0.172% are fraudelent transactions.
The features of the dataset include time (which is the amount of seconds elapsed since the first transaction),
transaction amount, and 28 unknown features which are the result of PCA performed on the original dataset to protect the privacy of the credit card owners.

## Data Split

The dataset was split to train and test sets with a ratio of 80:20. A validation set was not created as their was such little amount of fraud transactions.
Instead, cross-validation was used for the training of the classifier model. The test set was used to evaluate the final model.

## Data Cleaning

## Data Preparation

## Credits

The original dataset was collected and prepared by Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson, and Gianluca Bontempi.
The dataset is publicly available on Kaggle at https://www.kaggle.com/mlg-ulb/creditcardfraud.

## License

The dataset is licensed under the Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International.
It is available to use for non-commercial purposes with credit to the authors, and without modification.