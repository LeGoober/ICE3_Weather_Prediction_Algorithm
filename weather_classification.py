import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import os

# Data Cleaning and Preparation
weather_classification_data = pd.read_csv('weather_classification_data.csv')
weather_classification_data.columns = [
    "Temperature", "Humidity", "Wind Speed", "Precipitation (%)", "Cloud Cover",
    "Atmospheric Pressure", "UV Index", "Season", "Visibility (km)", "Location", "Weather Type"
]
weather_classification_data.drop("Atmospheric Pressure", axis=1, inplace=True)
subset_weather_data = weather_classification_data.iloc[:500, :]  # Increased to 500 samples
weather_classification_data_x = subset_weather_data.drop("Weather Type", axis=1)
weather_classification_data_y = subset_weather_data["Weather Type"]

# Convert categorical data to numerical
weather_classification_data_x = pd.get_dummies(weather_classification_data_x, columns=["Cloud Cover", "Location", "Season"])
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

    def entropy(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Add small value to avoid log(0)

    def gini_index(self, y):
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)

    def split(self, dataset, feature_index, threshold):
        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def information_gain(self, parent, left, right, criterion='entropy'):
        weight_left = len(left) / len(parent)
        weight_right = len(right) / len(parent)
        if criterion == 'entropy':
            return self.entropy(parent) - (weight_left * self.entropy(left) + weight_right * self.entropy(right))
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
                    current_info_gain = self.information_gain(parent, left_y, right_y, criterion='entropy')
                    if current_info_gain > max_info_gain:
                        best_split = {
                            'feature_index': feature_index,
                            'threshold': threshold,
                            'dataset_left': dataset_left,
                            'dataset_right': dataset_right,
                            'info_gain': current_info_gain
                        }
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
        return np.array([self.make_prediction(x, self.root) for x in X])

    def make_prediction(self, x, tree):
        if tree.value is not None:
            return tree.value
        if x[tree.feature_index] <= tree.threshold:
            return self.make_prediction(x, tree.left)
        return self.make_prediction(x, tree.right)

# Train and Evaluate with Cross-Validation
classifier = DecisionTreeClassifier(min_samples_split=3, max_depth=3)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=41)
cv_scores = []
for train_index, val_index in kf.split(X):
    X_train_cv, X_val_cv = X[train_index], X[val_index]
    y_train_cv, y_val_cv = y[train_index], y[val_index]
    classifier.fit(X_train_cv, y_train_cv)
    y_pred_cv = classifier.predict(X_val_cv)
    cv_scores.append(accuracy_score(y_val_cv, y_pred_cv))
print(f"Cross-Validation Scores: {cv_scores}")
print(f"Average CV Accuracy: {np.mean(cv_scores):.2f} (+/- {np.std(cv_scores) * 2:.2f})")

# Evaluation Metrics
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
print(f"Test Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")

# Create output directory
os.makedirs('output/figures', exist_ok=True)

# Visualizations
plt.figure(figsize=(10, 6))
plt.scatter(X_train[:, weather_classification_data_x.columns.get_loc("Temperature")],
            X_train[:, weather_classification_data_x.columns.get_loc("Humidity")],
            c=y_train, cmap='viridis')
plt.colorbar(label='Weather Type (Encoded)')
plt.xlabel('Temperature')
plt.ylabel('Humidity')
plt.title('Training Set: Temperature vs Humidity by Weather Type')
plt.savefig('output/figures/training_temperature_humidity_scatter.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(y_test, label='Actual Weather Type', marker='o')
plt.plot(y_pred, label='Predicted Weather Type', marker='x')
plt.xlabel('Test Sample Index')
plt.ylabel('Weather Type (Encoded)')
plt.title('Test Set: Actual vs Predicted Weather Types')
plt.legend()
plt.savefig('output/figures/test_actual_vs_predicted.png')
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(subset_weather_data.index[:500], subset_weather_data['Temperature'][:500], label='Temperature', color='blue')
plt.xlabel('Sample Index')
plt.ylabel('Temperature (°C)')
plt.title('Temperature Distribution Across Dataset')
plt.legend()
plt.savefig('output/figures/temperature_distribution.png')
plt.close()

# Activity Recommendation Mapping
weather_type_map = {0: "Sunny", 1: "Rainy", 2: "Snowy", 3: "Cloudy"}
activity_map = {
    "Sunny": ["Go to the beach", "Have a picnic", "Hike"],
    "Rainy": ["Stay indoors", "Watch a movie", "Read a book"],
    "Snowy": ["Build a snowman", "Go skiing", "Stay warm indoors"],
    "Cloudy": ["Visit a museum", "Go for a walk", "Do indoor crafts"]
}

# User Input Functions
def get_numeric_input(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input. Please enter a numeric value.")

def get_user_input():
    print("\nEnter weather conditions for activity recommendation or prediction:")
    temp = get_numeric_input("Temperature (°C): ")
    humid = get_numeric_input("Humidity (%): ")
    wind = get_numeric_input("Wind Speed (km/h): ")
    precip = get_numeric_input("Precipitation (%): ")
    uv = get_numeric_input("UV Index: ")
    vis = get_numeric_input("Visibility (km): ")
    cloud = input("Cloud Cover (clear/partly cloudy/overcast): ").strip().lower()
    loc = input("Location (e.g., coastal, inland): ").strip().lower()

    user_data = pd.DataFrame([[temp, humid, wind, precip, uv, vis]], 
                             columns=["Temperature", "Humidity", "Wind Speed", "Precipitation (%)", "UV Index", "Visibility (km)"])
    user_data["Cloud Cover"] = cloud
    user_data["Location"] = loc
    user_data = pd.get_dummies(user_data, columns=["Cloud Cover", "Location"])
    user_data = user_data.reindex(columns=weather_classification_data_x.columns, fill_value=0)
    return user_data.values

# Predict and Recommend Loop
while True:
    user_input = get_user_input()
    pred_encoded = classifier.predict(user_input)[0]
    pred_weather = weather_type_map.get(pred_encoded, "Unknown")
    activities = activity_map.get(pred_weather, ["No recommendation"])
    print(f"\nPredicted Weather: {pred_weather}")
    print("Suggested Activities:", ", ".join(activities))
    if input("\nWould you like to try again or exit? (yes/no): ").strip().lower() != 'yes':
        break