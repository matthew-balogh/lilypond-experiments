import os
from typing import Optional

def get_base_dir(file_path: str) -> str:
    """
    Determines the base directory of the project.

    Args:
        file_path (str): The path of the file calling this function.
                         In a notebook, you can often use `.`.
                         In a .py file, you should use `__file__`.

    Returns:
        str: The absolute path to the base directory.
    """
    return os.path.abspath(os.path.join(os.path.dirname(file_path), ".."))


def get_exports_dir(file_path: str, exports_name: str) -> str:
    """
    Creates and returns the path to an exports directory.

    Args:
        file_path (str): The path of the file calling this function.
                         In a notebook, you can often use `.`.
                         In a .py file, you should use `__file__`.
        exports_name (str): The name of the specific exports folder.

    Returns:
        str: The absolute path to the exports directory.
    """
    base_dir = get_base_dir(file_path)
    exports_dir = os.path.join(base_dir, "_exports", exports_name)
    os.makedirs(exports_dir, exist_ok=True)
    return exports_dir
