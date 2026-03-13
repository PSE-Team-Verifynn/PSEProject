# PSE Project: Neural Network Verification Algorithm Visualisation

## Installation and Execution
### Linux
Make sure that you have Python 3.13 (with pip, venv) and Git installed. Commands `python3.13`, `git` should be recognized by your system.
1. Download the latest linux release.
2. Unpack the downloaded `.tar.xz` archive
3. Run the `installer.sh` script and look out for any errors.
4. Start the program using the newly generated `start.sh` script.
5. Alternatively you can create the environment as described in the `ìnstaller.sh` manually (i. e. using a different pyton version).

### Windows
Make sure that you have Python 3.13 (with pip, venv) and Git installed. Commands `py -3.13`, `git` should be recognized by your system.
1. Download the latest linux release.
2. Unpack the downloaded `.zip` archive
3. Run the `installer.bat` script and look out for any errors.
4. Start the program using the newly generated `start.bat` script.
5. Alternatively you can create the environment as described in the `ìnstaller.bat` manually (i. e. using a different pyton version).

## Run custom algorithms
The program comes prepackaged with a base set of algorithms and test files. All algorithms stored in the `algorithms` directory will be recognized by the program.
Every algorithm consists of a single python file and contains a function with the following signature:
```py
def calculate_output_bounds(onnx_model, input_bounds: np.ndarray) -> np.ndarray:
```
To write your own algorithm, simply create a new python script that contains this function.
Additional libraries can be installed in the virtual python environment contained in the `venv` directory.

## How to get started with development
### Installation
To set up the project locally, follow these steps:
1. Clone the repo
2. Navigate to the project directory
3. Run `pip install -e .` to install dependencies (consider using a virtual environment)
4. Run `python -m src.nn_verification_visualisation` to start the app

### Testing
To run the provided tests, follow these steps:
1. Run `pip install -e .[dev]` to install development dependencies
2. Run `pytest` to run the tests

## Authors
- Alexander Mikhaylov
- Cedric Linde
- Elias Dörr
- Enrique Lopez
- Paul Schepperle

## Contact
Questions or feedback? Contact us at [verifynn@insert-greek-letter.de](mailto:verifynn@insert-greek-letter.de)
