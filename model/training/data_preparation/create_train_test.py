import os
from model.training.data_preparation.data_prep_functions import DataFrameManipulator, timing_decorator


@timing_decorator
def create_train_test(path_to_root, raw_from_root, raw_file_name):
    # DataFrame Manipulator
    data_man = DataFrameManipulator()

    # Path to raw data file
    file_path = os.path.join(path_to_root, raw_from_root, raw_file_name)

    # Import the data
    print("Raw:")
    df = data_man.import_csv(file_path)

    # Split the data into training and testing sets
    df_train, df_test = data_man.split(df, target_column='Class', stratify=True)

    # Path to raw data files for exports
    train_path = os.path.join(path_to_root, raw_from_root, 'train.csv')
    test_path = os.path.join(path_to_root, raw_from_root, 'test.csv')

    # Sort the sets by the 'Time' column and then export
    print("Training set:")
    df_train = data_man.sort(df_train, sort_column='Time')
    data_man.export_to_csv(df_train, train_path)
    print("File created - data/raw/train.csv")
    print("\n")

    print("Testing set:")
    df_test = data_man.sort(df_test, sort_column='Time')
    data_man.export_to_csv(df_test, test_path)
    print("File created - data/raw/test.csv")
    print("\n")


def main(path_to_root, raw_from_root, raw_file_name):
    create_train_test(path_to_root=path_to_root, raw_from_root=raw_from_root, raw_file_name=raw_file_name)


if __name__ == '__main__':
    main(path_to_root='../../..', raw_from_root='data/raw', raw_file_name='creditcard.csv')
