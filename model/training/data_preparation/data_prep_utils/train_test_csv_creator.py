import argparse
from model.training.data_preparation.data_prep_utils.dataframe_man import DataFrameManipulator


def train_test_csv_creator(
    csv_file_name: str,
    test_size: float,
    import_dir: str=None,
    export_dir: str=None,
    stratify_by: str=None,
    sort_by: str=None,
    sort_ascend: bool=True,
    reset_index: bool=False,
    reset_index_drop: bool=True,
    random_state: int=None
) -> None:
    """
    Import a csv file as a Pandas DataFrame, split it into train and test sets, and export both sets as CSV files.
    These files are called 'train.csv' and 'test.csv' and can be found in a new directory.

    :param csv_file_name:
        The name of the csv file to import.
    :param test_size:
        The portion of the original data that will be the test data.
    :param import_dir:
        The location path of the csv file to import. If not specified, uses current directory.
    :param export_dir:
        The location path of where to export the train and test files to. If not specified, same as import_dir.
    :param stratify_by:
        If the data split should be done stratified, specify the column name to stratify by. Default: None.
    : param sort_by:
        If the data should be sorted after the split, specify the column name to sort by. Default: None.
    :param sort_ascend:
        If the data should be sorted by ascending of the sort column when sort_by=True. Default: True.
    :param reset_index:
        If the returned data should have their index reset. Default: False.
    :param reset_index_drop:
        If the new index column from resetting index (if reset_index=True) should be dropped. Default: True.
    :param random_state:
        Specify for repeatablity. Default: None.
    """
    # DataFrameMan
    dataframe_man = DataFrameManipulator()

    # Create train and test CSV files.
    dataframe_man.export_train_test_csv(
        csv_file_name=csv_file_name,
        test_size=test_size,
        import_dir=import_dir,
        export_dir=export_dir,
        stratify_by=stratify_by,
        sort_by=sort_by,
        sort_ascend=sort_ascend,
        reset_index=reset_index,
        reset_index_drop=reset_index_drop,
        random_state=random_state
    )


def get_args() -> argparse.ArgumentParser:
    """
    Argument parser function for this file. Sets arguments and parses them for this file.

    :returns:
        Passed arguments.
    """
    # Argument parser
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument(
        '--csv_file_name', type=str, required=True,
        help='The name of the csv_file.'
        )
    parser.add_argument(
        '--test_size', type=float, required=True,
        help='The portion of the original data that will be the test data.'
        )
    parser.add_argument(
        '--import_dir', type=str, required=False, default=None,
        help='The path to the folder containing the csv file to import. If not specified, uses current directory.'
        )
    parser.add_argument(
        '--export_dir', type=str, required=False, default=None,
        help='The path to the folder for exporting the generated CSV files. If not specified, same as import_dir.'
        )
    parser.add_argument(
        '--stratify_by', type=str, required=False, default=None,
        help='If the data split should be done stratified, specify the column name to stratify by. Default: None.'
        )
    parser.add_argument(
        '--sort_by', type=str, required=False, default=None,
        help='If the data should be sorted after the split, specify the column name to sort by. Default: None.'
        )
    parser.add_argument(
        '--sort_ascend', type=bool, required=False, default=True,
        help='If the data should be sorted by ascending of the sort column when sort_by=True. Default: True.'
        )
    parser.add_argument(
        '--reset_index', type=bool, required=False, default=False,
        help='If the returned data should have their index reset. Default: False'
        )
    parser.add_argument(
        '--reset_index_drop', type=bool, required=False, default=True,
        help='If the new index column from resetting index (if reset_index=True) should be dropped. Default: True'
        )
    parser.add_argument(
        '--random_state', type=int, required=False, default=None,
        help='Specify for repeatablity. Default: None.'
        )
    
    # Parse args
    args = parser.parse_args()

    return args


def main() -> None:
    """
    Main entry point to create train and test files.
    This function splits a CSV file into train and test sets and exports them as CSV files.
    """
    # Get args
    args = get_args()

    # Create train test set
    train_test_csv_creator(
        csv_file_name=args.csv_file_name,
        test_size=args.test_size,
        import_dir=args.import_dir,
        export_dir=args.export_dir,
        stratify_by=args.stratify_by,
        sort_by=args.sort_by,
        sort_ascend=args.sort_ascend,
        reset_index=args.reset_index,
        reset_index_drop=args.reset_index_drop,
        random_state=args.random_state
    )


if __name__ == '__main__':
    main()
