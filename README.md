# PSE Project: Neural Network Verification Algorithm Visualisation

## How to run the program
The executables are packaged with every library needed to run the program. However, a lot of algorithms need external libraries to run. To set up the most common libraries, follow these steps:
1. Install Python 3.13 on your system
2. torch: pip install torch==2.8.0
3. auto_LiRPA: pip install git+"https://github.com/Verified-Intelligence/auto_LiRPA" (requires torch)

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
