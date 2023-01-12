import os
import pickle
from src.utils import timing_decorator
from model.training.modelling.model_classes.voting_classifier_model import VotingClassifierModel


@timing_decorator
def create(
    clf_path='../../classifiers'
) -> None:
    """
    Create the voting classifier model instance. Export the model as a pickle file.

    :param clf_path:
        The folder path to save the pickled model.
    """
    # Model instance
    clf = VotingClassifierModel(random_state=42)

    # Create directory if it doesn't exist
    os.makedirs(clf_path, exist_ok=True)

    # Save the model as a pickle file
    with open(str(os.path.join(clf_path, 'base_model.pkl')), 'wb') as f:
        pickle.dump(clf, f)


if __name__ == '__main__':
    create()
