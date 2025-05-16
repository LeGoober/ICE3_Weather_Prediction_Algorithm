Weather Classification Project Documentation
Overview
This project implements a Decision Tree Classifier to predict weather types (Sunny, Rainy, Snowy, Cloudy) using the weather_classification_data.csv dataset. It includes advanced evaluation metrics (k-fold cross-validation, confusion matrix, accuracy, precision, recall), an increased sample size of 500, entropy-based decision tree calculations, visualizations, and an interactive user interface for activity recommendations and custom predictions.

Objective

Primary Goal: Predict weather types based on features including Temperature, Humidity, Wind Speed, Precipitation (%), UV Index, Visibility (km), Cloud Cover, Season, and Location.
Secondary Goal: Recommend activities (e.g., "Go to the beach" for Sunny) and allow users to input custom weather conditions for predictions.
Dataset: weather_classification_data.csv with at least 500 samples used for improved accuracy.


Dependencies
The script uses the following Python libraries:

numpy: For numerical operations and array manipulations.
pandas: For data loading, cleaning, and preprocessing.
sklearn.model_selection.train_test_split: To split the dataset.
sklearn.model_selection.KFold: For k-fold cross-validation.
sklearn.metrics: For confusion matrix, accuracy, precision, and recall.
matplotlib.pyplot: For generating and saving visualizations.
os: To create directories for output files.


Data Processing
Steps

Load Dataset:

Loads weather_classification_data.csv using pandas.read_csv.
Columns: Temperature, Humidity, Wind Speed, Precipitation (%), Cloud Cover, Atmospheric Pressure, UV Index, Season, Visibility (km), Location, Weather Type.


Clean Data:

Drops Atmospheric Pressure as it is not used.
Uses the first 500 rows (instead of 100) to increase sample size and improve model accuracy.


Feature and Target Separation:

Features (weather_classification_data_x): All columns except Weather Type.
Target (weather_classification_data_y): Weather Type column.


Encode Categorical Data:

Converts categorical features (Cloud Cover, Location, Season) to numerical dummy variables using pd.get_dummies.
Encodes Weather Type into numerical values (e.g., Sunny=0, Rainy=1, Snowy=2, Cloudy=3) using astype('category').cat.codes.


Split Dataset:

Splits into training (80%) and test (20%) sets using train_test_split with a random state of 41 for reproducibility.




Decision Tree Algorithm
Core Guidelines
The Decision Tree Classifier is implemented from scratch with the following components:
1. Node Class

Represents a node in the decision tree.
Attributes:
feature_index: Index of the feature used for splitting.
threshold: Threshold value for the split.
left and right: Child nodes (left for ≤ threshold, right for > threshold).
info_gain: Information gain from the split (based on entropy or Gini).
value: Leaf node value (class label) if a leaf.



2. DecisionTreeClassifier Class

Initialization:

min_samples_split=2: Minimum samples to split a node.
max_depth=3: Maximum tree depth to control overfitting.


Gini Index (gini_index):

Measures impurity: ( \text{Gini} = 1 - \sum_{i=1}^{n} p_i^2 ), where ( p_i ) is the probability of class ( i ).


Entropy (entropy - newly added):

Measures impurity based on information theory: ( \text{Entropy} = - \sum_{i=1}^{n} p_i \log_2(p_i) ), where ( p_i ) is the proportion of class ( i ).
Used as an alternative to Gini for split evaluation.


Split Function (split):

Splits dataset into left (≤ threshold) and right (> threshold) subsets based on a feature and threshold.


Information Gain (information_gain):

Calculates gain using entropy or Gini.
Formula (with entropy): ( \text{Info Gain} = \text{Entropy(parent)} - (\text{weight}{\text{left}} \cdot \text{Entropy(left)} + \text{weight}{\text{right}} \cdot \text{Entropy(right)}) ).
weight_left and weight_right are proportions of samples.


Best Split (get_best_split):

Evaluates all features and thresholds, selecting the split with maximum information gain.
Returns feature index, threshold, and resulting datasets.


Tree Building (build_tree):

Recursively builds the tree.
Stops if:
Samples < min_samples_split.
Depth > max_depth.
No positive information gain.


Creates a leaf with the majority class (np.bincount(y).argmax()) if stopping criteria are met.


Fit (fit):

Combines features and labels, initiates tree construction.


Predict (predict):

Traverses the tree for each sample to predict the class.


Make Prediction (make_prediction):

Recursively follows the tree based on feature values and thresholds.



3. Training and Evaluation

Trains on the training set and evaluates on the test set.
Incorporates k-fold cross-validation for robustness.


Evaluation Metrics
1. K-Fold Cross-Validation

Uses 5-fold cross-validation to assess model performance across different data splits.
Calculates average accuracy across folds to reduce variance.

2. Confusion Matrix

Generates a matrix to show true positives, false positives, true negatives, and false negatives for each weather type.
Helps identify misclassifications (e.g., Rainy predicted as Cloudy).

3. Accuracy

Computed as: ( \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Predictions}} ).
Reported as a percentage on the test set.

4. Precision

Calculated as: ( \text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}} ).
Measures the proportion of positive predictions that are correct.

5. Recall

Calculated as: ( \text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}} ).
Measures the proportion of actual positives correctly identified.


Output Diagrams
Saved in output/figures/ directory.
1. Training Set Visualization

File: training_temperature_humidity_scatter.png
Type: Scatter Plot
Description: Plots Temperature vs. Humidity for the training set, colored by Weather Type (encoded). Visualizes data distribution for decision tree splits.

2. Test Set Visualization

File: test_actual_vs_predicted.png
Type: Line Graph
Description: Compares actual vs. predicted Weather Types for the test set, showing model performance.

3. Feature Distribution Visualization

File: temperature_distribution.png
Type: Line Graph
Description: Plots Temperature trend across the dataset, highlighting patterns for tree splits.


Activity Recommendation
Mapping

Weather Type Mapping: {0: "Sunny", 1: "Rainy", 2: "Snowy", 3: "Cloudy"}.
Activity Mapping:
Sunny: "Go to the beach", "Have a picnic", "Hike".
Rainy: "Stay indoors", "Watch a movie", "Read a book".
Snowy: "Build a snowman", "Go skiing", "Stay warm indoors".
Cloudy: "Visit a museum", "Go for a walk", "Do indoor crafts".



User Interaction

Input: Users enter Temperature, Humidity, Wind Speed, Precipitation, UV Index, Visibility, Cloud Cover, and Location.
Prediction: Model predicts weather type and suggests activities.
Loop: Continues until the user chooses to exit.

Custom Prediction Input

Users can input weather parameters to test predictions manually, with the model providing the predicted weather type.


Execution Flow

Data Processing: Load and preprocess 500 samples.
Model Training: Train with k-fold cross-validation.
Evaluation: Compute accuracy, precision, recall, and confusion matrix.
Visualizations: Generate and save the three plots.
User Interaction: Prompt for activity recommendations and custom predictions in a loop.


Notes

Sample Size: Increased to 500 for better accuracy.
Entropy: Added as an impurity measure alongside Gini.
Evaluation: Enhanced with cross-validation and detailed metrics.
Execution: Run with python weather_classification.py after ensuring sklearn.metrics is installed (pip install scikit-learn).

