from src.utils import timing_decorator
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from model.training.modelling.model_classes.voting_classifier_model import VotingClassifierModel
from imblearn.pipeline import Pipeline


class PipelineModel:
    """
    A pipeline odel that includes:
    Robust Scaler
    Random UnderSampler
    Tomek Links
    Voting Classifier(SVM+KNN+Bagging(LogReg))
    """
    def __init__(self, random_state: int=None):
        self.scaler = RobustScaler()
        self.rus = RandomUnderSampler(random_state=random_state)
        self.tl = TomekLinks()
        self.clf = VotingClassifierModel(random_state=random_state)
        self.pipeline = Pipeline([('scaler', self.scaler), ('rus', self.rus), ('tl', self.tl), ('clf', self.clf)])

    @timing_decorator
    def fit(self, X, y) -> object:
        """
        Fit the pipeline model on the input data and labels.

        :param X:
            The input data: array-like, shape(n_samples, n_features).
        :param y:
            The labels for the input data: array-like, shape (n_samples,)

        :returns:
            An instance of self.
        """
        self.pipeline.fit(X, y)
        return self

    @timing_decorator
    def predict(self, X):
        """
        Predict the labels for the input data using the pipeline.

        :param X:
            The input data: array-like, shape (n_samples, n_features).

        :returns:
            The predicted labels for the input data: array-like, shape (n_samples,)
        """
        return self.pipeline.predict(X)
    
    @timing_decorator
    def predict_proba(self, X):
        """
        Predict the class probabilities for the input data using the pipeline model.

        :param X:
            The input data: array-like, shape (n_samples, n_features).

        :returns:
            The class probabilities for the input data: array-like, shape (n_samples, n_classes).
        """
        return self.pipeline.predict_proba(X)
