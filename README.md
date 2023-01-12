# Credit Card Fraud Detection

This project aims to develop a model for detecting credit card fraud using the credit card fraud detection dataset from Kaggle. The dataset is imbalanced and the performance metrics focused on were recall to detect as many fraud transactions as possible, even if non-fraud transactions were misclassified (of course to an acceptable degree). Specificity and accuracy were also looked at, as they gave a good indication of if non-fraud transactions were being classified correctly.

## Data

The dataset represents a common problem of an imbalanced dataset.
It contains 284,807 transactions, of which 0.172% are fraudelent transactions.
The features of the dataset include time (which is the amount of seconds elapsed since the first transaction),
transaction amount, and 28 unknown features which are the result of PCA performed on the original dataset to protect the privacy of the credit card owners.

![Imbalanced dataset](https://github.com/abuhasan12/credit-card-fraud-detection/blob/main/readme_imgs/Class%20Imbalance.png)

## Data Analysis

The first step in the project was to perform a thorough analysis of the data. This included removing extreme outliers to see the impact on the model's ability to classify fraud transactions. Resampling was performed on the data and multiple methods of undersampling were tried including random undersampling, cluster centroid undersampling and these in combination with Tomek links. It was found that Tomek links followed by random undersampling performed the best. Correlations between features were also analyzed. Dimensionality reduction methods such as PCA, t-SNE, MDS and LLE were used to visualize the data and classes better. The t-SNE visualisation looked the best.

## Model

Different models were compared including base logistic regression, SVM, decision tree, Naive Bayes and k-NN. Naive Bayes performed badly and was dropped. The remaining models were fine-tuned to see how they would perform. Bagging classifiers of k-NN, logistic regression and SVM, as well as random forest (and extra trees) were used for ensemble methods. Boosting methods such as ADA, gradient and XGBoost were also tried. 

The chosen model was a **voting classifier** created using scikit-learn. It achieved a consisting of:
* A support-vector-machine classifier
* A KNN classifier
* A Bagged Logstic Regression classifier

The metrics used to evaluate the validation performance of the final classifier were recall and specificity:
* Recall - to assess the model's performance in classifying fraud transactions
* Specificity - to assess the model's performance in classifying non-fraud transactions

The final model had a recall of 95%, accuracy of 96% and specificity of 96% on the test set.

## Conclusion

In this project, a model for credit card fraud detection was developed using the credit card fraud detection dataset from Kaggle. The model was based on a voting classifier of SVM, k-NN and bagging logistic regression models and achieved a high recall of 95%, accuracy of 96% and specificity of 96% on the test set.

To read more about the problem investigation and development, please refer to the notebook provided.