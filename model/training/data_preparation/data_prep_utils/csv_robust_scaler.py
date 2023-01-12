import argparse
from model.training.data_preparation.data_prep_utils.dataframe_man import DataFrameManipulator


def csv_robust_scaler(
    csv_file_name: str,
    scale_columns: list,
    import_dir: str=None,
    export_dir: str=None,
    sort_by: str=None,
    sort_ascend: bool=True,
    reset_index: bool=False,
    reset_index_drop: bool=True
) -> None:
    """
    Import a csv file as a Pandas DataFrame,
    Scale features using SKLearn's RobustScaler,
    Export the new DataFrame as a csv file in the export directory in 'scaled'.

    :param csv_file_name:
        The name of the csv file to import.
    :param scale_columns:
        List of column names to scale.
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
    """
    # DataFrameMan
    dataframe_man = DataFrameManipulator()

    # Scale data and export scaled data.
    dataframe_man.robust_scale_csv(
        csv_file_name=csv_file_name,
        scale_columns=scale_columns,
        import_dir=import_dir,
        export_dir=export_dir,
        sort_by=sort_by,
        sort_ascend=sort_ascend,
        reset_index=reset_index,
        reset_index_drop=reset_index_drop
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
        '--scale_columns', nargs='*', type=str, required=True,
        help='A list of column(s) to apply robust scaling too.'
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

    # Parse args
    args = parser.parse_args()

    return args


def main() -> None:
    """
    Main entry point to create train and test files.
    This function scales a csv file on the specified columns.
    """
    # Get args
    args = get_args()

    # Scale the data
    csv_robust_scaler(
        csv_file_name=args.csv_file_name,
        scale_columns=args.scale_columns,
        import_dir=args.import_dir,
        export_dir=args.export_dir,
        sort_by=args.sort_by,
        sort_ascend=args.sort_ascend,
        reset_index=args.reset_index,
        reset_index_drop=args.reset_index_drop
    )


if __name__ == '__main__':
    main()
