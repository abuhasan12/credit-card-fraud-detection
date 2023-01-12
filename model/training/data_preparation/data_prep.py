from src.utils import timing_decorator
from model.training.data_preparation.data_prep_process import create_train_test_sets, clean_training_data, process_training_data


@timing_decorator
def data_prep(
    raw_path = '../../../data/raw',
    train_test_path = '../../../data/raw/train_test',
    clean_path = '../../../data/clean',
    processed_path = '../../../data/processed',
) -> None:
    """
    Python script file to create train and test sets from raw file,
    Clean the data by performing extreme outlier removal,
    Process the data by scaling features and undersampling,
    Exporting the final processed dataset as a CSV file.
    """
    # Prep
    create_train_test_sets.create(raw_path=raw_path)
    clean_training_data.clean(train_test_path=train_test_path, clean_path=clean_path)
    process_training_data.process(clean_path=clean_path, processed_path=processed_path)


if __name__ == '__main__':
    data_prep()
