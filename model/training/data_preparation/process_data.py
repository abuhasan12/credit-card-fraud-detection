import os
from model.training.data_preparation.data_prep_functions import DataFrameManipulator, create_directory, timing_decorator


@timing_decorator
def process_data(path_to_root, clean_from_root, clean_file_name, processed_from_root):
    # DataFrame Manipulator
    data_man = DataFrameManipulator()

    # Path to raw data file
    file_path = os.path.join(path_to_root, clean_from_root, clean_file_name)

    # Import the data
    print("Cleaned dataset:")
    df = data_man.import_csv(file_path)

    # Scale the unscaled features
    df = data_man.scale_columns(df, column_names=['Amount', 'Time'])

    # Undersample the data using Random Undersampling and TomekLinks
    print("Scaled dataset:")
    df = data_man.undersample_rus_tl(df, target='Class')

    # Sort by 'Time'
    print("Resampled dataset:")
    df = data_man.sort(df, sort_column='Time')

    # Path to processed data folder for exports
    processed_directory = os.path.join(path_to_root, processed_from_root)

    # Create the directory if it does not exist
    create_directory(processed_directory)

    # Path to processed data file for exports
    processed_path = os.path.join(path_to_root, processed_from_root, 'processed_train.csv')

    # Export as new file
    data_man.export_to_csv(df, processed_path)
    print("File created - data/processed/processed_train.csv")
    print("\n")


def main(path_to_root, clean_from_root, clean_file_name, processed_from_root):
    process_data(path_to_root=path_to_root, clean_from_root=clean_from_root, clean_file_name=clean_file_name,
            processed_from_root=processed_from_root)


if __name__ == '__main__':
    main(path_to_root='../../..', clean_from_root='data/clean', clean_file_name='clean_train.csv',
         processed_from_root='data/processed')
