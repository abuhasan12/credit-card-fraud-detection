import time
import os
from sklearn.preprocessing import RobustScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from imblearn.under_sampling import TomekLinks


def timing_decorator(func):
    """
    Times the execution of a function and prints the elapsed time.

    Parameters:
    func (function): the function to be timed

    Returns:
    function: the decorated function
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} took {end - start:.4f} seconds to run')
        return result
    return wrapper

def print_custom_string(custom_string):
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(custom_string)
            result = func(*args, **kwargs)
            print("Done.\n")
            return result
        return wrapper
    return decorator

def print_dataframe_info(target_column, name):
    """
    Prints the number of rows and the number of rows for each unique value in the target column of one or more pandas dataframes.

    Parameters:
    target_column (str): name of the column to count unique values in
    *dataframe_names (str): names of the dataframes to print info for. If not provided, dataframe names will default to 'Dataframe 1', 'Dataframe 2', etc.

    Returns:
    function: decorated function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            df = func(*args, **kwargs)
            rows, cols = df.shape
            print(f'{name} has {rows} rows and {cols} columns')
            unique_values = df[target_column].nunique()
            print(f'{target_column} has {unique_values} unique values')
            for value, count in df[target_column].value_counts().items():
                print(f'{value}: {count} rows')
            print("\n")
            return df

        return wrapper

    return decorator


def print_dataframes_info(target_column, names):
    """
    Prints the number of rows and the number of rows for each unique value in the target column of one or more pandas dataframes.

    Parameters:
    target_column (str): name of the column to count unique values in
    *dataframe_names (str): names of the dataframes to print info for. If not provided, dataframe names will default to 'Dataframe 1', 'Dataframe 2', etc.

    Returns:
    function: decorated function
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            df1, df2 = func(*args, **kwargs)
            dfs = [df1, df2]
            for name, df in zip(names, dfs):
                rows, cols = df.shape
                print(f'{name} has {rows} rows and {cols} columns')
                unique_values = df[target_column].nunique()
                print(f'{target_column} has {unique_values} unique values')
                for value, count in df[target_column].value_counts().items():
                    print(f'{value}: {count} rows')
                print("\n")
            return df1, df2

        return wrapper

    return decorator


class DataFrameManipulator:
    """
    A class for dealing with dataframes.
    """
    @print_custom_string('Importing data.')
    @print_dataframe_info('Class', 'Imported Dataset')
    def import_csv(self, filepath) -> pd.DataFrame:
        """
        Import data from a CSV file and return it as a Pandas DataFrame.

        Parameters:
        filepath (str): The path to the CSV file.

        Returns:
        pandas.DataFrame: A DataFrame containing the data from the CSV file.
        """
        df = pd.read_csv(filepath)
        return df

    @print_custom_string('Splitting data into train and test sets.')
    @print_dataframes_info('Class', ['Training Dataset', 'Testing Dataset'])
    def split(self, df, target_column, test_size=0.2, stratify=False, random_state=42):
        """
        Split the data into training and testing sets.

        Parameters:
        df (pandas.DataFrame): The data to be split.
        target_column (str): The name of the target column in the data.

        Returns:
        tuple: A tuple containing the training and testing sets as two separate DataFrames.
        """

        X = df.drop(columns=[target_column])
        y = df[target_column]

        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=test_size, random_state=random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        df_train = pd.concat([X_train, y_train], axis=1)
        df_test = pd.concat([X_test, y_test], axis=1)

        return df_train, df_test

    @print_custom_string('Sorting dataframe.')
    def sort(self, df, sort_column, ascending=True):
        """
        Sort the data by the sort_column.

        Parameters:
        df (pandas.DataFrame): The data to be split.
        sort_column (str): The name of the column to sort by.

        Returns:
        tuple: A pandas DataFrames.
        """
        if not ascending:
            return df.sort_values(by=sort_column, ascending=False).reset_index(drop=True)
        else:
            return df.sort_values(by=sort_column).reset_index(drop=True)

    @print_custom_string('Scaling the data.')
    def scale_columns(self, df: pd.DataFrame, column_names: list) -> pd.DataFrame:
        """
        Scale the specified columns of a DataFrame using RobustScaler.

        Parameters:
        - df: The DataFrame to scale.
        - column_names: A list of column names to scale.

        Returns:
        A copy of the input DataFrame with the specified columns scaled.
        """
        scaler = RobustScaler()
        df_scaled = df.copy()
        df_scaled[column_names] = scaler.fit_transform(df_scaled[column_names])
        return df_scaled

    @print_custom_string('Resampling the data.')
    @print_dataframe_info('Class', 'Resampled Dataset')
    def undersample_rus_tl(self, df: pd.DataFrame, target) -> pd.DataFrame:
        """
        Apply random undersampling and Tomek links to a DataFrame.

        Parameters:
        - df: The DataFrame to undersample.

        Returns:
        A copy of the input DataFrame that has been undersampled.
        """
        X = df.drop(target, axis=1)
        y = df[target]

        rus = RandomUnderSampler(random_state=42)
        rus_X, rus_y = rus.fit_resample(X, y)

        tl = TomekLinks()
        tl_X, tl_y = tl.fit_resample(rus_X, rus_y)

        df_undersampled = pd.concat([tl_X, tl_y], axis=1)

        return df_undersampled

    @print_custom_string('Exporting the data.')
    @print_dataframe_info('Class', 'Exported Dataset')
    def export_to_csv(self, df, filepath):
        """
        Export the data as a CSV file.

        Parameters:
        df (pandas.DataFrame): The data to be exported.
        filepath (str): The path to the file where the data should be exported.
        """
        df.to_csv(filepath, index=False)
        return df


def create_directory(directory: str) -> None:
    """
    Create the specified directory if it does not exist.

    Parameters:
    - directory: The path to the directory to create.

    Returns:
    None
    """
    if not os.path.exists(directory):
        if os.path.isfile(directory):
            raise ValueError(f'{directory} is a file, not a directory')
        else:
            os.makedirs(directory)