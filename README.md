# ICE3_Weather_Prediction_Algorithm
Weather-Based Activity Recommender
Overview
This project implements a Decision Tree machine learning algorithm to classify weather conditions using the weather_classification_data.csv dataset. Based on the predicted weather type (Sunny, Rainy, Snowy, Cloudy), the application recommends appropriate activities (e.g., "Go to the beach" for Sunny, "Stay indoors" for Rainy). The project is designed for group collaboration, utilizing Jupyter notebooks for exploration and Python scripts for modular, reusable code.


Functionality
>The main functionalities of the project is that we are focusing on the **"Weather-Based Activity Predictor."** The goal of this project is to predict the best outdoor activity based on simple weather conditions.
### **How it works:**
- The input features might include:
    - Temperature (Hot, Mild, Cold)
    - Rain (Yes, No)        
    - Wind Speed (High, Low)
- The **decision tree** would classify activities such as:
    - If **hot & no rain**, then **go for a picnic**.   
    - If **cold & high wind**, then **stay indoors**.
    - If **mild & no rain**, then **go jogging**.



Data Processing: Loads and cleans the weather dataset, handling missing values, outliers, and encoding categorical features (e.g., Cloud Cover, Season, Location).
Model Training: Trains a Decision Tree Classifier using scikit-learn to predict weather types based on features like Temperature, Humidity, Wind Speed, etc.
Activity Recommendation: Maps predicted weather types to activity suggestions:
Sunny: Go to the beach, have a picnic, or hike.
Rainy: Stay indoors, watch a movie, or read a book.
Snowy: Build a snowman, go skiing, or stay warm indoors.
Cloudy: Visit a museum, go for a walk, or do indoor crafts.


Visualization: Generates plots, including the Decision Tree structure and performance metrics (e.g., confusion matrix, accuracy).
Collaboration: Organized structure with notebooks for exploration and scripts for production code, suitable for team contributions.

Project Structure
weather_activity_recommender/
│
├── data/
│   ├── raw/
│   │   └── weather_classification_data.csv  # Raw dataset
│   ├── processed/
│   │   └── cleaned_weather_data.csv        # Cleaned dataset
│
├── notebooks/
│   ├── 01_data_exploration.ipynb           # Data analysis and visualization
│   ├── 02_model_training.ipynb             # Model training and evaluation
│   ├── 03_activity_recommendation.ipynb    # Activity recommendation testing
│
├── src/
│   ├── __init__.py                        # Makes src a Python package
│   ├── data_cleaning.py                   # Data preprocessing functions
│   ├── feature_engineering.py             # Feature extraction/encoding
│   ├── decision_tree_model.py             # Decision Tree training/prediction
│   ├── activity_recommender.py            # Activity suggestion logic
│   ├── evaluation.py                      # Model evaluation metrics
│   └── visualization.py                   # Plotting functions
│
├── tests/
│   ├── test_data_cleaning.py              # Tests for data cleaning
│   ├── test_decision_tree.py              # Tests for model
│
├── output/
│   ├── figures/                           # Plots (e.g., Decision Tree)
│   ├── models/                            # Saved models
│   └── reports/                           # Performance reports
│
├── .gitignore                             # Ignores venv, .env, outputs
├── requirements.txt                       # Project dependencies
├── README.md                              # This file
└── main.py                                # Main pipeline script

Setup Instructions

Clone the Repository:
git clone <repository-url>
cd weather_activity_recommender


Create and Activate Virtual Environment:
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows


Install Dependencies:
pip install -r requirements.txt

Required packages: numpy, pandas, scikit-learn, matplotlib, seaborn, jupyter.

Place Dataset:

Copy weather_classification_data.csv into data/raw/.


Run the Pipeline:
python main.py

This loads the dataset, cleans it, trains the Decision Tree, and outputs predictions with activity recommendations.

Explore Notebooks:

Open Jupyter Notebook:jupyter notebook


Navigate to notebooks/ and run:
01_data_exploration.ipynb for data insights.
02_model_training.ipynb for model training.
03_activity_recommendation.ipynb for testing activity suggestions.





Usage

Run the Full Pipeline:

Execute main.py to process data, train the model, and generate activity recommendations.
Outputs (plots, models, reports) are saved in output/.


Interactive Exploration:

Use notebooks/01_data_exploration.ipynb to visualize data distributions (e.g., Temperature vs. Weather Type).
Use notebooks/02_model_training.ipynb to experiment with Decision Tree parameters (e.g., max_depth).
Use notebooks/03_activity_recommendation.ipynb to test activity suggestions for specific weather conditions.


Example Command-Line Prediction:
from src.decision_tree_model import predict_weather
from src.activity_recommender import suggest_activity

# Example input (replace with actual values)
weather_data = {
    "Temperature": 25.0,
    "Humidity": 60,
    "Wind Speed": 5.0,
    "Precipitation (%)": 10.0,
    "Cloud Cover": "clear",
    "Atmospheric Pressure": 1015.0,
    "UV Index": 6,
    "Season": "Summer",
    "Visibility (km)": 8.0,
    "Location": "coastal"
}
weather_type = predict_weather(weather_data)
activity = suggest_activity(weather_type)
print(f"Weather: {weather_type}, Suggested Activity: {activity}")



Dependencies
Listed in requirements.txt:

numpy: Numerical computations
pandas: Data manipulation
scikit-learn: Decision Tree algorithm
matplotlib, seaborn: Visualization
jupyter: Interactive notebooks
(Optional) python-dotenv: For environment variables

Contributing

Create a new branch for your feature: git checkout -b feature-name.
Write tests in tests/ for new functions.
Commit changes and create a pull request.
Clear Jupyter notebook outputs before committing:jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace notebooks/*.ipynb

=======
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
>>>>>>> 0b77dbdab8b0851f6a0f7e85495ed3315369b5a7


Notes

<<<<<<< HEAD
The dataset (weather_classification_data.csv) contains features like Temperature, Humidity, and Weather Type, used to train the Decision Tree.
Ensure .env (if used) is added to .gitignore for sensitive data.
Outputs (e.g., Decision Tree plots, confusion matrices) are saved in output/figures/.

Contact
For questions, contact the team via the repository's issue tracker.
=======
Sample Size: Increased to 500 for better accuracy.
Entropy: Added as an impurity measure alongside Gini.
Evaluation: Enhanced with cross-validation and detailed metrics.
Execution: Run with python weather_classification.py after ensuring sklearn.metrics is installed (pip install scikit-learn).

>>>>>>> 0b77dbdab8b0851f6a0f7e85495ed3315369b5a7
