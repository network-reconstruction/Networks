# network_geo

`network_geo` is a Python package designed to handle network-related geographical data. This package provides various utilities, modules, and submodules to simplify the processing and analysis of such data.

## File Tree

Below is the file tree structure of the `network_geo` package:


network_geo/
├── network_geo/
│   ├── __init__.py
│   ├── submodule1/
│   │   ├── __init__.py
│   │   ├── module1.py
│   │   └── sub_submodule1/
│   │       ├── __init__.py
│   │       └── sub_module1.py
│   ├── submodule2/
│   │   ├── __init__.py
│   │   └── module2.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_submodule1/
│   │   ├── __init__.py
│   │   ├── test_module1.py
│   │   └── test_sub_submodule1/
│   │       ├── __init__.py
│   │       └── test_sub_module1.py
│   ├── test_submodule2/
│   │   ├── __init__.py
│   │   └── test_module2.py
│   └── test_utils/
│       ├── __init__.py
│       └── test_helpers.py
├── setup.py
├── README.md
└── .gitignore



## Installation

To install the `network_geo` package, use the following command:

```bash
pip install network_geo 
```

## Usage

Here's a quick guide on how to use the network_geo package.

### Importing Modules

You can import and use functions or classes from the package as follows:

```python
from network_geo.submodule1.module1 import function_in_module1
from network_geo.submodule1.sub_submodule1.sub_module1 import function_in_sub_module1
from network_geo.submodule2.module2 import function_in_module2
from network_geo.utils.helpers import helper_function

# Example usage
result1 = function_in_module1()
result2 = function_in_sub_module1()
result3 = function_in_module2()
helper_result = helper_function()
```
### Running Tests

To run the tests for the package, you can use the unittest framework:

```bash
python -m unittest discover -s tests
```

## Contributing

Please ask Yi Yao Tan (yao-creative) before contributing. If you'd like to contribute to network_geo, please follow these steps:

    1) Fork the repository.
    2) Create a new branch (git checkout -b feature-branch).
    3) Make your changes.
    4) Commit your changes (git commit -m 'Add new feature').
    5) Push to the branch (git push origin feature-branch).
    6) Open a pull request.

NOTE: PLEASE COMMENT AND TYPE ALL OF THE FUNCTIONS, PERFORM ERROR HANDLING TO THE BEST OF YOUR ABILITY 

### Example of well written function

```python
from typing import List

def calculate_average(numbers: List[float]) -> float:
    """
    Calculate the average of a list of numbers.

    Args:
        numbers (List[float]): A list containing numeric values.

    Returns:
        float: The average value of the numbers.

    Raises:
        ValueError: If the input list `numbers` is empty.

    Example:
        >>> calculate_average([1, 2, 3, 4, 5])
        3.0
    """
    if not numbers:
        raise ValueError("Input list cannot be empty")

    return sum(numbers) / len(numbers)
  
```

### Init files

#### `__init__.py`

```python
# network_geo/submodule1/__init__.py

# Import specific functions/classes for submodule1
from .module1 import function_in_module1
from .sub_submodule1.sub_module1 import function_in_sub_module1

# Optionally define what gets imported when someone imports the submodule
__all__ = [
    'function_in_module1',
    'function_in_sub_module1',
]

# Optional initialization code
print(f"Initializing submodule1: {__name__}")

# You can also define variables or perform other setup tasks here
# For example:
# VERSION = '1.0'

# End of __init__.py
```

### File Contents for Example Functions

Here are some example functions you can use to populate the modules and submodules:

#### `module1.py`

```python
def function_in_module1():
    return "Function in module1"
```
#### `sub_module1.py`

```python
def function_in_sub_module1():
    return "Function in sub_submodule1"
```

#### `helpers.py`

```python
def helper_function():
    return "Helper function"
```


### Example Test Files

#### `test_module1.py`

```python
import unittest
from network_geo.submodule1.module1 import function_in_module1

class TestModule1(unittest.TestCase):
    def test_function_in_module1(self):
        self.assertEqual(function_in_module1(), "Function in module1")

if __name__ == '__main__':
    unittest.main()
```

#### `test_sub_module1.py`

```python
import unittest
from network_geo.submodule1.sub_submodule1.sub_module1 import function_in_sub_module1

class TestSubModule1(unittest.TestCase):
    def test_function_in_sub_module1(self):
        self.assertEqual(function_in_sub_module1(), "Function in sub_submodule1")

if __name__ == '__main__':
    unittest.main()
```

####  `test_helpers.py`

```python
import unittest
from network_geo.utils.helpers import helper_function

class TestHelpers(unittest.TestCase):
    def test_helper_function(self):
        self.assertEqual(helper_function(), "Helper function")

if __name__ == '__main__':
    unittest.main()
```



## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contact
For any questions or inquiries, please contact ```yytanacademic@gmail.com```