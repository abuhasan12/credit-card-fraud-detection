import os
import pickle
import pandas
from model.training.data_preparation.data_prep_utils.dataframe_man import DataFrameManipulator


class ModelManipulator:
    """
    A class for performing functions on models.
    """
    def fit_export_pkl(
        self,
        model_name: str,
        features: pandas.DataFrame,
        target: pandas.Series=None,
        import_dir: str=None,
        export_dir: str=None
    ) -> None:
        """
        Import a pickle file as a model,
        Fit the model using feature and target variables,
        Export the model as a pickle file.

        :param model_name:
            Name of model to fit.
        :param X:
            Feature variables.
        :param y:
            Target variable.
        :param import_dir:
            The directory path where the base model is. If not specified, will use current directory.
        :param export_dir:
            The directory path to save the fitted model to. If not specified, same as import_dir.
        """
        # Base model
        with open(str(os.path.join(import_dir, model_name)), 'rb') as f:
            model = pickle.load(f)

        # Fit model
        if type(target) == type(None):
            model.fit(features.values)
        else:
            model.fit(features.values, target.values.ravel())

        # Export path
        if not export_dir:
            export_dir = import_dir

        # Save the model as a pickle file
        with open(str(os.path.join(export_dir, 'fitted_model.pkl')), 'wb') as f:
            pickle.dump(model, f)

    def fit_csv_export_pkl(
        self,
        model_name: str,
        data_file: str,
        data_location: str=None,
        target_variable: str=None,
        import_dir: str=None,
        export_dir: str=None
    ) -> None:
        """
        Import a csv file as a pandas DataFrame,
        Import a pickle file as a model,
        Fit the data from the csv file to th model,
        Export the model.

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
        # DataFrameMan
        dataframe_man = DataFrameManipulator()

        # Import csv file as DataFrame
        df = dataframe_man.csv_to_df(csv_file_name=data_file, import_dir=data_location)

        # X and y
        X, y = dataframe_man.extract_features_target(df=df, target_variable=target_variable)

        # Import model, fit X and y, then export.
        self.fit_export_pkl(
            model_name=model_name,
            features=X,
            target=y,
            import_dir=import_dir,
            export_dir=export_dir
        )