from src.utils import timing_decorator
from model.training.data_preparation.data_prep_utils.train_test_csv_creator import train_test_csv_creator


@timing_decorator
def create(
    raw_path = '../../../../data/raw'
) -> None:
    """
    Function to create the train and test CSV files from the raw creditcard.csv file.
    
    :param raw_path:
        Raw data folder path.
    """
    # Create train and test sets
    train_test_csv_creator(
        csv_file_name='creditcard.csv',
        test_size=0.2,
        import_dir=raw_path,
        stratify_by='Class',
        sort_by='Time',
        reset_index=True,
        random_state=42
    )


if __name__ == '__main__':
    create()
