import os
import pandas
from src.utils import timing_decorator
from model.training.data_preparation.data_prep_utils.csv_robust_scaler import csv_robust_scaler
from model.training.data_preparation.data_prep_utils.csv_random_undersampler import csv_random_undersampler
from model.training.data_preparation.data_prep_utils.csv_tomek_links_undersampler import csv_tomek_links_undersampler


@timing_decorator
def process(
    clean_path = '../../../../data/clean',
    processed_path = '../../../../data/processed'
) -> None:
    """
    Function the process the training data for modelling.

    :param clean_path:
        Folder path for clean training data.
    :param processed_path:
        Folder path for processed data.
    """
    # Scale the data using Robust Scaler.
    csv_robust_scaler(
        csv_file_name='clean_train.csv',
        scale_columns=['Time', 'Amount'],
        import_dir=clean_path,
        export_dir=processed_path,
        sort_by='Time',
        reset_index=True
    )

    # Undersample the data using Random Undersampling.
    csv_random_undersampler(
        csv_file_name='scaled.csv',
        target_variable='Class',
        import_dir=str(os.path.join(processed_path, 'robust_scaled')),
        export_dir=processed_path,
        sort_by='Time',
        reset_index=True,
        random_state=42
    )

    # Undersample the data using TomekLinks.
    csv_tomek_links_undersampler(
        csv_file_name='random_undersampled.csv',
        target_variable='Class',
        import_dir=str(os.path.join(processed_path, 'rus')),
        export_dir=processed_path,
        sort_by='Time',
        reset_index=True
    )

    # Processed data
    df = pandas.read_csv(str(os.path.join(processed_path, 'tl/tomeklinks_undersampled.csv')))
    # Insert code to read if needed.
    df.to_csv(str(os.path.join(processed_path, 'processed_train.csv')), index=False)


if __name__ == '__main__':
    process()
