import os
from src.utils import timing_decorator
from model.training.data_preparation.data_prep_utils.dataframe_man import DataFrameManipulator
from model.training.data_preparation.data_prep_utils.data_prep_functions import print_dataframe_info


@timing_decorator
@print_dataframe_info('Clean Training Dataset', classification=True)
def clean(
    train_test_path = '../../../../data/raw/train_test',
    clean_path = '../../../../data/clean'
) -> None:
    """
    Function to clean the creditcard training dataset.

    :param train_test_path:
        Folder path for training dataset.
    :param clean_path:
        Folder path for clean training data.
    """
    # DataFrameMan
    dataframe_man = DataFrameManipulator()

    # Importing training data CSV as DataFrame
    df = dataframe_man.csv_to_df(csv_file_name='train.csv', import_dir=train_test_path)

    # Remove extreme outliers from non-fraud transactions
    print("Removing extreme outliers.")
    df = df.loc[((df['Class'] == 0) & (df['Amount'] <= 1010.875)) | df['Class'] == 1]
    print("Done.")
    print("\n")

    # Export cleaned data as CSV
    os.makedirs(clean_path, exist_ok=True)
    df.to_csv(str(os.path.join(clean_path, 'clean_train.csv')), index=False)

    return df


if __name__ == '__main__':
    clean()
