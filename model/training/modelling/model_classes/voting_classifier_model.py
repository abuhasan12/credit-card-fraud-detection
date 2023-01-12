from src.utils import timing_decorator
from sklearn.ensemble import BaggingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


class VotingClassifierModel:
    """
    A class that creates a voting classifier model consisting of an SVM classifier, a KNN classifier,
    and a bagging classifier.
    """
    def __init__(self, random_state: int=None):
        self.svm_clf = SVC(C=100,
                           degree=2,
                           gamma='auto',
                           random_state=random_state)
        self.knn_clf = KNeighborsClassifier(n_neighbors=1,
                                            p=1)
        self.bagging_estimator = LogisticRegression(C=100,
                                                         max_iter=500,
                                                         random_state=random_state)
        self.bagging_clf = BaggingClassifier(estimator=self.bagging_estimator,
                                             max_features=0.5,
                                             max_samples=0.75,
                                             random_state=random_state)
        self.voting_clf = VotingClassifier(estimators=[('svm', self.svm_clf),
                                                       ('knn', self.knn_clf),
                                                       ('bagging', self.bagging_clf)])

    @timing_decorator
    def fit(self, X, y) -> object:
        """
        Fit the voting classifier model on the input data and labels.

        :param X:
            The input data: array-like, shape(n_samples, n_features).
        :param y:
            The labels for the input data: array-like, shape (n_samples,)

        :returns:
            An instance of self.
        """
        self.voting_clf.fit(X, y)
        return self

    @timing_decorator
    def predict(self, X):
        """
        Predict the labels for the input data using the voting classifier model.

        :param X:
            The input data: array-like, shape (n_samples, n_features).

        :returns:
            The predicted labels for the input data: array-like, shape (n_samples,)
        """
        return self.voting_clf.predict(X)

    @timing_decorator
    def predict_proba(self, X):
        """
        Predict the class probabilities for the input data using the voting classifier model.

        :param X:
            The input data: array-like, shape (n_samples, n_features).

        :returns:
            The class probabilities for the input data: array-like, shape (n_samples, n_classes).
        """
        return self.voting_clf.predict_proba(X)
