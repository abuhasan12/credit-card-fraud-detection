import argparse
from model.training.data_preparation.data_prep_utils.dataframe_man import DataFrameManipulator


def csv_random_undersampler(
    csv_file_name: str,
    target_variable :str,
    import_dir: str=None,
    export_dir: str=None,
    sort_by: str=None,
    sort_ascend: bool=True,
    reset_index: bool=False,
    reset_index_drop: bool=True,
    random_state: int=None
) -> None:
    """
    Import a csv file as a Pandas DataFrame,
    Random Undersample the data using imblearn's RandomUnderSampler,
    Exports the new DataFrame as a csv file in the export directory in 'rus'
    
    param csv_file_name:
        The name of the csv file to import.
    :param target_variable:
        The target feature for undersampling.
    :param import_dir:
        The location path of the csv file to import. If not specified, uses current directory.
    :param export_dir:
        The location path of where to export the scaled file. If not specified, same as import_dir.
    :param sort_by:
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

    # Scale data and export scaled data.
    dataframe_man.random_undersample_csv(
        csv_file_name=csv_file_name,
        target_variable=target_variable,
        import_dir=import_dir,
        export_dir=export_dir,
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
        help='The path to the folder containing the csv file to import.'
    )
    parser.add_argument(
        '--target_variable', type=str, required=True,
        help='The feature to undersample by.'
        )
    parser.add_argument(
        '--import_dir', type=str, required=False, default=None,
        help='The path to the folder containing the csv file to import. If not specified, uses current directory.'
        )
    parser.add_argument(
        '--export_dir', type=str, required=False, default=None,
        help='The path to the folder for exporting the generated CSV file. If not specified, same as import_dir.'
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
    This function underamples a csv file on the target feature using random undersampling.
    """
    # Get args
    args = get_args()

    # Scale the data
    csv_random_undersampler(
        csv_file_name=args.csv_file_name,
        target_variable=args.target_variable,
        import_dir=args.import_dir,
        export_dir=args.export_dir,
        sort_by=args.sort_by,
        sort_ascend=args.sort_ascend,
        reset_index=args.reset_index,
        reset_index_drop=args.reset_index_drop,
        random_state=args.random_state
    )


if __name__ == '__main__':
    main()
