import argparse
from model.training.modelling.modelling_utils.model_man import ModelManipulator


def csv_fitter(
    model_name: str,
    data_file: str,
    data_location: str=None,
    target_variable: str=None,
    import_dir: str=None,
    export_dir: str=None
) -> None:
    """
    Trains a model imported from a pickle file using data in a csv file.

    :param model_name:
        The name of the base model to be fitted.
    :param data_file:
        The name of the csv data file to fit the base model with.
    :param data_location:
        The path to the csv data file to fit the base model with.
    :param target_variable:
        Target feature for fitting if applicable, otherwise None.
    :param import_dir:
        The directory path where the base model is. If not specified, will use current directory.
    :param export_dir:
        The directory path to save the fitted model to. If not specified, same as import_dir.
    """
    # ModelMan
    model_man = ModelManipulator()

    # Fit csv file to model.
    model_man.fit_csv_export_pkl(
        model_name=model_name,
        data_file=data_file,
        data_location=data_location,
        target_variable=target_variable,
        import_dir=import_dir,
        export_dir=export_dir
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
        '--model_name', type=str, required=True,
        help='The name of the model file.'
        )
    parser.add_argument(
        '--data_file', type=str, required=True,
        help='The csv file name.'
        )
    parser.add_argument(
        '--data_location', type=str, required=False, default=None,
        help='The csv data file path. If not specified, uses current directory.'
        )
    parser.add_argument(
        '--target_variable', type=str, required=False, default=None,
        help='Target feature of the data if applicable. Default: None.'
        )
    parser.add_argument(
        '--import_dir', type=str, required=False, default=None,
        help='The path to the folder containing the model to fit. If not specified, uses current directory.'
        )
    parser.add_argument(
        '--export_dir', type=str, required=False, default=None,
        help='The path to the folder for exporting the fitted model. If not specified, same as import_dir.'
        )
    
    # Parse args
    args = parser.parse_args()

    return args


def main() -> None:
    """
    Main entry point to fit a base model using a csv file for data and pickle file for model.
    The fitted model is also exported.
    """
    # Get args
    args = get_args()

    # Fit and export model
    csv_fitter(
        model_name=args.model_name,
        data_file=args.data_file,
        data_location=args.data_location,
        target_variable=args.target_variable,
        import_dir=args.import_dir,
        export_dir=args.export_dir
    )


if __name__ == '__main__':
    main()