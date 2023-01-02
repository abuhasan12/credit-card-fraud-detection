import pandas as pd
from sklearn.model_selection import train_test_split


class DataFrameManipulator:
    """
    A class for splitting, sorting, and exporting data.
    """
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

    def export_to_csv(self, df, filepath):
        """
        Export the data as a CSV file.

        Parameters:
        df (pandas.DataFrame): The data to be exported.
        filepath (str): The path to the file where the data should be exported.
        """
        df.to_csv(filepath, index=False)