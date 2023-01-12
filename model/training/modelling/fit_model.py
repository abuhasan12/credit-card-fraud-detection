from src.utils import timing_decorator
from model.training.modelling.modelling_utils.csv_fitter import csv_fitter

@timing_decorator
def fit(
    clf_path = '../../classifiers',
    data_location = '../../../data/processed'
) -> None:
    """
    Function to fit the voting clf imported from a pickle file and export it.
    """
    # Fit and export model
    csv_fitter(
        model_name='base_model.pkl',
        data_file='processed_train.csv',
        data_location=data_location,
        target_variable='Class',
        import_dir=clf_path
    )


if __name__ == '__main__':
    fit()