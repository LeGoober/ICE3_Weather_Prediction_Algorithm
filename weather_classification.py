import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Data Cleaning and Preparation
weather_classification_data = pd.read_csv('weather_classification_data.csv')  # Updated path
weather_classification_data.columns = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Precipitation (%)",
    "Cloud Cover",
    "Atmospheric Pressure",
    "UV Index",
    "Season",
    "Visibility (km)",
    "Location",
    "Weather Type"
]

new_weather_classification_data = weather_classification_data.drop("Atmospheric Pressure", axis=1)
subset_weather_data = new_weather_classification_data.iloc[:100, :]
weather_classification_data_x = subset_weather_data.drop("Weather Type", axis=1)
weather_classification_data_y = subset_weather_data["Weather Type"]

# Convert categorical data to numerical where applicable
weather_classification_data_x = pd.get_dummies(weather_classification_data_x, columns=["Cloud Cover", "Location"])
weather_classification_data_y = weather_classification_data_y.astype('category').cat.codes

# Split data
X = weather_classification_data_x.values
y = weather_classification_data_y.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=41)

# Decision Tree Classifier
class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, info_gain=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.info_gain = info_gain
        self.value = value

class DecisionTreeClassifier():
    def __init__(self, min_samples_split=2, max_depth=3):
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def gini_index(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, left, right):
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        return self.gini_index(parent) - (weight_left * self.gini_index(left) + weight_right * self.gini_index(right))

    def get_best_split(self, dataset, num_samples, num_features):
        best_split = {}
        max_info_gain = -float('inf')
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            thresholds = np.unique(feature_values)
            for threshold in thresholds:
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    parent = dataset[:, -1]
                    left_y = dataset_left[:, -1]
                    right_y = dataset_right[:, -1]
                    current_info_gain = self.information_gain(parent, left_y, right_y)
                    if current_info_gain > max_info_gain:
                        best_split['feature_index'] = feature_index
                        best_split['threshold'] = threshold
                        best_split['dataset_left'] = dataset_left
                        best_split['dataset_right'] = dataset_right
                        best_split['info_gain'] = current_info_gain
                        max_info_gain = current_info_gain
        return best_split

    def build_tree(self, dataset, current_depth=0):
        X, y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = X.shape
        if num_samples >= self.min_samples_split and current_depth <= self.max_depth:
            best_split = self.get_best_split(dataset, num_samples, num_features)
            if best_split and best_split['info_gain'] > 0:
                left_subtree = self.build_tree(best_split['dataset_left'], current_depth + 1)
                right_subtree = self.build_tree(best_split['dataset_right'], current_depth + 1)
                return Node(best_split['feature_index'], best_split['threshold'], left_subtree, right_subtree, best_split['info_gain'])
        leaf_value = np.bincount(y.astype(int)).argmax()
        return Node(value=leaf_value)

    def fit(self, X, y):
        dataset = np.concatenate((X, y.reshape(-1, 1)), axis=1)
        self.root = self.build_tree(dataset)

    def predict(self, X):
        predictions = [self.make_prediction(x, self.root) for x in X]
        return np.array(predictions)

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature_index]
        if feature_value <= tree.threshold:
            return self.make_prediction(x, tree.left)
        return self.make_prediction(x, tree.right)

# Train and Evaluate
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")