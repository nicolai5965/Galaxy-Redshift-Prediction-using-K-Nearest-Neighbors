# Galaxy-Redshift-Prediction-using-K-Nearest-Neighbors
Galaxy Redshift Prediction using K-Nearest Neighbors

This repository contains an implementation of the K-Nearest Neighbors (KNN) algorithm for galaxy redshift prediction using the provided galaxies_train.csv and galaxies_test.csv datasets.

## Data Preparation
* Load the train and test datasets using pandas.
* Extract the features and target variables from the train and test datasets.
* Split the test dataset into test and validation sets using the train_test_split function.

## KNN Implementation
The KNearestNeighbors class implements the KNN algorithm, with the following parameters:

* n_neighbors: The number of neighbors to use for regression (default: 3).
* metric: The distance metric to use ('euclidean', 'manhattan', or 'minkowski') (default: 'euclidean').
* p: The power parameter for the Minkowski distance (required when metric is 'minkowski').
* weighted: Whether to use distance-weighted voting (default: False).

The class provides the following methods:

* fit(): Train the KNN model on the provided data.
* predict_instance(): Predict the redshift for a single instance.
* predict(): Predict the redshifts for a batch of instances.

## Model Building and Evaluation
* Find the best value of k based on the Root Mean Squared Error (RMSE) for the validation set.
* Train KNN models with the best k values for each distance metric (euclidean, manhattan, and minkowski) on the train dataset.
* Make predictions on the test dataset and compute the RMSE for each distance metric.
* Calculate additional evaluation metrics, including Mean Absolute Error (MAE), R-squared, Mean Squared Logarithmic Error (MSLE), and Mean Absolute Percentage Error (MAPE).
## Usage
To use this code, simply run the provided Python script in Google Colab. Ensure that you have the required libraries installed, including NumPy and pandas, and have access to the galaxies_train.csv and galaxies_test.csv datasets in your Google Drive.

## Conclusion
This implementation demonstrates the effectiveness of the KNN algorithm for galaxy redshift prediction using various distance metrics. The best k value and RMSE are reported for each distance metric, and the model with the lowest RMSE can be selected as the most accurate. Additional evaluation metrics provide further insights into the model's performance.

