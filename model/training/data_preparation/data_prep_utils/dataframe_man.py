import os
from model.training.data_preparation.data_prep_utils.data_prep_functions import print_dataframe_info, print_dataframes_info
import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks


class DataFrameManipulator:
    """
    A class for dealing with dataframes.
    """
    def extract_features_target(
        self,
        df: pandas.DataFrame,
        target_variable: str
    ):
        """
        Function to return features and target from DataFrame.

        :param df:
            DataFrame to extract from.
        :param target_variable:
            Target column for y.

        :returns:
            Tuple containing X and y.
        """
        # X and y
        X = df.drop(target_variable, axis=1)
        y = df[target_variable]

        return X, y

    @print_dataframe_info(name='Imported CSV', classification=True)
    def csv_to_df(
        self,
        csv_file_name: str,
        import_dir: str=None
    ) -> pandas.DataFrame:
        """
        Import a csv file as a pandas DataFrame.

        :param csv_file_name:
            The name of the csv file to import.
        :param import_dir:
            The location path of the csv file to import. If not specified, uses current directory.

        :returns:
            Pandas DataFrame from the imported CSV file.
        """
        if import_dir:
            csv_file_path = str(os.path.join(import_dir, csv_file_name))
        else:
            csv_file_path = csv_file_name
        # Import csv
        print("Importing CSV file as DataFrame...")
        df = pandas.read_csv(csv_file_path)
        print("Done.\n")

        return df

    @print_dataframes_info(names=['Training Set', 'Test Set'], classification=True)
    def csv_to_train_test_df(
        self,
        csv_file_name: str,
        test_size: float,
        import_dir: str=None,
        stratify_by: str=None,
        sort_by: str=None,
        sort_ascend: bool=True,
        reset_index: bool=False,
        reset_index_drop: bool=True,
        random_state: int=None
    ) -> pandas.DataFrame:
        """
        Given the location of a csv file,
        Import it as a Pandas DataFrame,
        Split it to train and test DataFrames and return them.

        :param csv_file_name:
            The name of the csv file to import.
        :param test_size:
            The portion of the original data that will be the test data.
        :param import_dir:
            The location path of the csv file to import. If not specified, uses current directory.
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

        :returns:
            A tuple containing train and test DataFrames.
        """
        # CSV to df
        df = self.csv_to_df(csv_file_name=csv_file_name, import_dir=import_dir)

        # Create train and test DataFrames
        print("Creating train and test set...")
        if stratify_by:
            stratify_by = df[stratify_by]
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, stratify=stratify_by)
        print("Done.\n")
        
        # Sort and reset index if needed
        if sort_by:
            train_df = train_df.sort_values(by=sort_by, ascending=sort_ascend)
            test_df = test_df.sort_values(by=sort_by, ascending=sort_ascend)
        if reset_index:
            train_df = train_df.reset_index(drop=reset_index_drop)
            test_df = test_df.reset_index(drop=reset_index_drop)

        return train_df, test_df

    def export_train_test_csv(
        self,
        csv_file_name: str,
        test_size: float,
        import_dir: str=None,
        export_dir:str=None,
        stratify_by: str=None,
        sort_by: str=None,
        sort_ascend: bool=True,
        reset_index: bool=False,
        reset_index_drop: bool=True,
        random_state: int=None
    ) -> None:
        """
        Import a csv file as a Pandas DataFrame, split it into train and test sets,
        Sort and reset the index of the sets if needed,
        Then export both sets as CSV files.
        These files are called 'train.csv' and 'test.csv' and are exported to a new directory.

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
        # Import CSV file as DataFrame and split to train and test DataFrames
        train_df, test_df = self.csv_to_train_test_df(
            csv_file_name=csv_file_name,
            test_size=test_size,
            import_dir=import_dir,
            stratify_by=stratify_by,
            sort_by=sort_by,
            sort_ascend=sort_ascend,
            reset_index = reset_index,
            reset_index_drop=reset_index_drop,
            random_state=random_state
        )
        
        # Paths to export
        if not export_dir:
            export_dir = import_dir
        os.makedirs(os.path.join(export_dir, 'train_test'), exist_ok=True)
        train_csv = os.path.join(export_dir, 'train_test', 'train.csv')
        test_csv = os.path.join(export_dir, 'train_test', 'test.csv')
        
        # Export as CSVs
        print("Exporting train and test set as CSV files...")
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        print("Done.\n")

    @print_dataframe_info(name='Scaled DataFrame', classification=True)
    def robust_scale_df(
        self,
        df: pandas.DataFrame,
        scale_columns: list,
        sort_by: str=None,
        sort_ascend: bool=True,
        reset_index: bool=False,
        reset_index_drop: bool=True
    ) -> pandas.DataFrame:
        """
        Scale the features of a DataFrame using SKLearn's RobustScaler.

        :param df:
            DataFrame to scale.
        :param scale_columns:
            List of column names to scale.
        :param sort_by:
            If the data should be sorted after the split, specify the column name to sort by. Default: None.
        :param sort_ascend:
            If the data should be sorted by ascending of the sort column when sort_by=True. Default: True.
        :param reset_index:
            If the returned data should have their index reset. Default: False.
        :param reset_index_drop:
            If the new index column from resetting index (if reset_index=True) should be dropped. Default: True.

        :returns:
            Scaled DataFrame.
        """
        # RobustScaler
        scaler = RobustScaler()

        # Create copy
        scaled_df = df.copy()

        # Scale columns of the df
        scaled_df[scale_columns] = scaler.fit_transform(df[scale_columns])
        
        # Sort and reset index if needed
        if sort_by:
            scaled_df = scaled_df.sort_values(by=sort_by, ascending=sort_ascend)
        if reset_index:
            scaled_df = scaled_df.reset_index(drop=reset_index_drop)

        return scaled_df

    def robust_scale_csv(
        self,
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
        # CSV to df
        df = self.csv_to_df(csv_file_name=csv_file_name, import_dir=import_dir)

        # Scale df
        scaled_df = self.robust_scale_df(
            df=df,
            scale_columns=scale_columns,
            sort_by=sort_by,
            sort_ascend=sort_ascend,
            reset_index=reset_index,
            reset_index_drop=reset_index_drop
        )
        
        # Paths to export
        if not export_dir:
            export_dir = import_dir
        os.makedirs(os.path.join(export_dir, 'robust_scaled'), exist_ok=True)
        scaled_csv = os.path.join(export_dir, 'robust_scaled', 'scaled.csv')
        
        # Export as CSVs
        print("Exporting scaled data as CSV files...")
        scaled_df.to_csv(scaled_csv, index=False)
        print("Done.\n")

    @print_dataframe_info(name='Random Undersampled DataFrame', classification=True)
    def random_undersample_df(
        self,
        df: pandas.DataFrame,
        target_variable: str,
        sort_by: str=None,
        sort_ascend: bool=True,
        reset_index: bool=False,
        reset_index_drop: bool=True,
        random_state: int=None
    ) -> pandas.DataFrame:
        """
        Undersample the data using imblearn's RandomUnderSampler.

        :param df:
            DataFrame to scale.
        :param target_variable:
            The target feature for undersampling.
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

        :returns:
            Random Undersampled DataFrame.
        """
        # X and y
        X, y = self.extract_features_target(df=df, target_variable=target_variable)

        # Random Undersampling
        rus = RandomUnderSampler(random_state=random_state)
        rus_X, rus_y = rus.fit_resample(X, y)
        
        # Create DataFrame
        rus_df = pandas.concat([rus_X, rus_y], axis=1)
        
        # Sort and Index if needed
        if sort_by:
            rus_df = rus_df.sort_values(by=sort_by, ascending=sort_ascend)
        if reset_index:
            rus_df = rus_df.reset_index(drop=reset_index_drop)

        return rus_df

    def random_undersample_csv(
        self,
        csv_file_name: str,
        target_variable: str,
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
        Undrsample the data using RandomUnderSampler,
        Export the new DataFrame as a csv file in the export directory in 'rus'.

        :param csv_file_name:
            The name of the csv file to import.
        :param target_variable:
            The target feature for undersampling.
        :param import_dir:
            The location path of the csv file to import. If not specified, uses current directory.
        :param export_dir:
            The location path of where to export the undersampled file. If not specified, same as import_dir.
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
        # CSV to df
        df = self.csv_to_df(csv_file_name=csv_file_name, import_dir=import_dir)

        # Random undersample df
        rus_df = self.random_undersample_df(
            df=df,
            target_variable=target_variable,
            sort_by=sort_by,
            sort_ascend=sort_ascend,
            reset_index=reset_index,
            reset_index_drop=reset_index_drop,
            random_state=random_state
        )
        
        # Paths to export
        if not export_dir:
            export_dir = import_dir
        os.makedirs(os.path.join(export_dir, 'rus'), exist_ok=True)
        rus_csv = os.path.join(export_dir, 'rus', 'random_undersampled.csv')
        
        # Export as CSV
        print("Exporting random undersampled data as CSV file...")
        rus_df.to_csv(rus_csv, index=False)
        print("Done.\n")

    @print_dataframe_info(name='TomekLinks Undesampled DataFrame', classification=True)
    def tomek_links_undersample_df(
        self,
        df: pandas.DataFrame,
        target_variable: str,
        sort_by: str=None,
        sort_ascend: bool=True,
        reset_index: bool=False,
        reset_index_drop: bool=True
    ) -> pandas.DataFrame:
        """
        Undersample the data using imblearn's TomekLinks.

        :param df:
            DataFrame to scale.
        :param target_variable:
            The target feature for undersampling.
        :param sort_by:
            If the data should be sorted after the split, specify the column name to sort by. Default: None.
        :param sort_ascend:
            If the data should be sorted by ascending of the sort column when sort_by=True. Default: True.
        :param reset_index:
            If the returned data should have their index reset. Default: False.
        :param reset_index_drop:
            If the new index column from resetting index (if reset_index=True) should be dropped. Default: True.

        :returns:
            Random Undersampled DataFrame.
        """
        # X and y
        X, y = self.extract_features_target(df=df, target_variable=target_variable)

        # TomekLinks
        tl = TomekLinks()
        tl_X, tl_y = tl.fit_resample(X, y)
        
        # Create DataFrame
        tl_df = pandas.concat([tl_X, tl_y], axis=1)
        
        # Sort and Index if needed
        if sort_by:
            tl_df = tl_df.sort_values(by=sort_by, ascending=sort_ascend)
        if reset_index:
            tl_df = tl_df.reset_index(drop=reset_index_drop)

        return tl_df

    def tomek_links_undersample_csv(
        self,
        csv_file_name: str,
        target_variable: str,
        import_dir: str=None,
        export_dir: str=None,
        sort_by: str=None,
        sort_ascend: bool=True,
        reset_index: bool=False,
        reset_index_drop: bool=True
    ) -> None:
        """
        Import a csv file as a Pandas DataFrame,
        Undrsample the data using RandomUnderSampler,
        Export the new DataFrame as a csv file in the export directory in 'rus'.

        :param csv_file_name:
            The name of the csv file to import.
        :param target_variable:
            The target feature for undersampling.
        :param import_dir:
            The location path of the csv file to import. If not specified, uses current directory.
        :param export_dir:
            The location path of where to export the undersampled file. If not specified, same as import_dir.
        :param sort_by:
            If the data should be sorted after the split, specify the column name to sort by. Default: None.
        :param sort_ascend:
            If the data should be sorted by ascending of the sort column when sort_by=True. Default: True.
        :param reset_index:
            If the returned data should have their index reset. Default: False.
        :param reset_index_drop:
            If the new index column from resetting index (if reset_index=True) should be dropped. Default: True.
        """
        # CSV to df
        df = self.csv_to_df(csv_file_name=csv_file_name, import_dir=import_dir)

        # Random undersample df
        tl_df = self.tomek_links_undersample_df(
            df=df,
            target_variable=target_variable,
            sort_by=sort_by,
            sort_ascend=sort_ascend,
            reset_index=reset_index,
            reset_index_drop=reset_index_drop
        )
        
        # Paths to export
        if not export_dir:
            export_dir = import_dir
        os.makedirs(os.path.join(export_dir, 'tl'), exist_ok=True)
        tl_csv = os.path.join(export_dir, 'tl', 'tomeklinks_undersampled.csv')
        
        # Export as CSV
        print("Exporting tomek links undersampled data as CSV file...")
        tl_df.to_csv(tl_csv, index=False)
        print("Done.\n")

