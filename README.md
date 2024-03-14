# CiFake Image Classification
### This repository contains a Python script for training a convolutional neural network (CNN) to classify images from the CiFake dataset as either real or AI-generated synthetic images.

## Dataset
### The CiFake dataset is available at [/kaggle/input/cifake-real-and-ai-generated-synthetic-images/](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images). It contains two directories, train and test, each containing subdirectories with real and synthetic images.

## Requirements
### The following libraries and modules are required to run the script:

* Python 3.x
* TensorFlow
* Keras
* Pandas
* NumPy
* Matplotlib
* keras_tuner

## Usage
* Ensure that the required libraries are installed.
* Run the script using a Python environment or a Jupyter Notebook.


## Code Overview
> The script uses Version 3 of the [CIFake Dataset](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images/versions/3)

The script performs the following steps:

1. Imports the necessary libraries and modules.
2. Sets up a TensorBoard callback for logging and visualization during training.
3. Loads the training and test datasets from the provided paths.
4. Defines helper functions for data preprocessing and normalization.
5. Sets up early stopping, learning rate reduction, and model checkpointing callbacks.
6. Defines custom Keras layers for MBConv6 and MBConv1.
7. Defines a build_model function that uses the Keras Tuner library to create and compile a CNN model with various hyperparameters that can be tuned.
8. Initializes a RandomSearch tuner object from Keras Tuner and sets up the search space for hyperparameter tuning.
9. Runs the hyperparameter search, training the models on the provided training and validation datasets, and using the specified callbacks.
10. Retrieves and prints a summary of the best performing model from the tuner's search.
11. Saves the best model to a file.

## Results
The script outputs the summary of the best performing model and saves it to a file named BEST<TIMESTAMP>.keras, where <TIMESTAMP> is the current date and time.

## Contributing
If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request.

## License
See LICENSE.
