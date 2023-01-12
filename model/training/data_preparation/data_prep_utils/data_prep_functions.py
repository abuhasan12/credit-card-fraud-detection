def print_dataframe_info(
    name: str, classification: bool=False
):
    """
    Prints the number of rows and columns
    and the number of rows for each unique value in the target column of one a pandas dataframe.

    :param str name:
        Name of the dataframs to print info for.
    :param bool classification:
        If the dataframe is for classification, to count unique values in.

    :returns:
        The decorated function.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            df = func(*args, **kwargs)
            rows, cols = df.shape
            print(f'{name} has {rows} rows and {cols} columns')
            if classification:
                class_column = df.columns[-1]
                unique_values = df[class_column].nunique()
                print(f'{class_column} has {unique_values} unique values')
                for value, count in df[class_column].value_counts().items():
                    print(f'{value}: {count} rows')
            print("\n")
            return df

        return wrapper

    return decorator


def print_dataframes_info(
    names: list, classification: bool=False
):
    """
    Prints the number of rows and columns
    and the number of rows for each unique value in the target column of two pandas dataframes.

    :param list names:
        Names of the dataframes to print info for.
    :param str classification:
        If the dataframe is for classification, to count unique values in.

    :returns:
        The decorated function.
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            df1, df2 = func(*args, **kwargs)
            dfs = [df1, df2]
            for name, df in zip(names, dfs):
                rows, cols = df.shape
                print(f'{name} has {rows} rows and {cols} columns')
                if classification:
                    class_column = df.columns[-1]
                    unique_values = df[class_column].nunique()
                    print(f'{class_column} has {unique_values} unique values')
                    for value, count in df[class_column].value_counts().items():
                        print(f'{value}: {count} rows')
                print("\n")
            return df1, df2

        return wrapper

    return decorator
