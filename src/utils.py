import time
import os


def timing_decorator(func):
    """
    Times the execution of a function and prints the elapsed time.

    :param func:
        The function to be timed.

    :returns:
        The decorated function.
    """
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'function took {end - start:.4f} seconds to run')
        return result
    return wrapper


def print_custom_string(custom_string: str):
    """
    A decorator that prints a custom string before and after calling the decorated
    function.

    :param custom_string:
        The string to be printed.

    :returns:
        The decorated function.
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(custom_string)
            result = func(*args, **kwargs)
            print("Done.\n")
            return result
        return wrapper
    return decorator


def find_file(filename: str, search_dir: str) -> str:
    """
    Check if a file exists in the current directory. If not, search for it in
    another folder within the same directory and return the path of the file if
    it exists.

    :param filename:
        The name of the file to search for.
    :param search_dir:
        The name of the folder to search in if the file is not found in the
        current directory.

    :returns:
        The path of the file if it is found, or an empty string if the file is
        not found.
    """
    # Check if the file exists in the current directory
    if os.path.exists(filename):
        return filename

    # If the file is not found in the current directory, search for it in
    # the specified directory
    search_path = os.path.join(search_dir, filename)
    if os.path.exists(search_path):
        return search_path

    # If the file is not found, return an empty string
    return ""