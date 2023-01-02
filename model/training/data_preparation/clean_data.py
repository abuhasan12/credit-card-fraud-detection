import os
from model.training.data_preparation.data_prep_functions import DataFrameManipulator, create_directory, timing_decorator


@timing_decorator
def clean_data(path_to_root, raw_from_root, train_file_name, clean_from_root):
    # DataFrame Manipulator
    data_man = DataFrameManipulator()

    # Path to raw data file
    file_path = os.path.join(path_to_root, raw_from_root, train_file_name)

    # Import the data
    print("Training dataset:")
    df = data_man.import_csv(file_path)

    # Remove extreme outliers from non-fraud transactions
    print("Removing extreme outliers.")
    df = df.loc[((df['Class'] == 0) & (df['Amount'] <= 1010.875)) | df['Class'] == 1]
    print("Done.")
    print("\n")

    # Sort by 'Time'
    print("Cleaned dataset:")
    df = data_man.sort(df, sort_column='Time')

    # Path to clean data folder for exports
    clean_directory = os.path.join(path_to_root, clean_from_root)

    # Create the directory if it does not exist
    create_directory(clean_directory)

    # Path to clean data file for exports
    clean_path = os.path.join(path_to_root, clean_from_root, 'clean_train.csv')

    # Export as new file
    data_man.export_to_csv(df, clean_path)
    print("File created - data/clean/clean_train.csv")
    print("\n")


def main(path_to_root, raw_from_root, train_file_name, clean_from_root):
    clean_data(path_to_root=path_to_root, raw_from_root=raw_from_root, train_file_name=train_file_name,
               clean_from_root=clean_from_root)


if __name__ == '__main__':
    main(path_to_root='../../..', raw_from_root='data/raw', train_file_name='train.csv',
         clean_from_root='data/clean')
