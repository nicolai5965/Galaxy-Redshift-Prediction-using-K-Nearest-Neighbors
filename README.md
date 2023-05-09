# Galaxy Redshift Prediction using K-Nearest Neighbors

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

## Potential Improvements
### Efficiency Improvements
The current implementation can be slow for large datasets. Using optimized data structures, such as KD-trees or Ball-trees, can significantly speed up the search for the nearest neighbors. These tree-based data structures can reduce the computational complexity of finding the nearest neighbors from O(n) to O(log n) in many cases.

### Vectorization
The current implementation relies on Python loops for distance calculations and neighbor search, which can be slow. Utilizing vectorization techniques with NumPy can help speed up the calculations by taking advantage of low-level optimizations and parallelism. This can lead to substantial performance gains, especially when working with large datasets.

### Visualization
Visualizing the dataset, the decision boundaries, or the performance metrics can help provide a better understanding of how KNN works and how it is affected by different hyperparameters. Some possible visualizations include:

* Scatter plots of the input features, colored by the target variable (redshift), to reveal patterns and relationships within the data.
* Decision boundaries for different values of k or distance metrics, which can help illustrate how the model generalizes across the feature space.
* Learning curves that plot the RMSE (or other performance metrics) against different values of k or the size of the training dataset, which can help identify the optimal model complexity and training set size.
Incorporating these visualizations can help identify potential areas for improvement, such as feature selection or transformation, and provide insights into the model's performance and behavior.

## Conclusion
This implementation demonstrates the effectiveness of the KNN algorithm for galaxy redshift prediction using various distance metrics. The best k value and RMSE are reported for each distance metric, and the model with the lowest RMSE can be selected as the most accurate. Additional evaluation metrics provide further insights into the model's performance.  

