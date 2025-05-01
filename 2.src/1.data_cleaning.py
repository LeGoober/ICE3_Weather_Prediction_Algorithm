#1. Importing the pandas libaray for data manipulation
import pandas as pd
#2. Unpacking the dataset
weather_classification_data = pd.read_csv('../data/raw_data/weather_classification_data.csv')

weather_classification_data.columns = [
    "Temperature",
    "Humidity",
    "Wind Speed",
    "Precipitation (%)",
    "Cloud Cover",
    "Atmospheric Pressure",
    "UV Index",
    "Visibility (km)",
    "Location",
    "Weather Type"
]

# 3. Remove the 'Atmospheric Pressure' column
new_weather_classification_data = weather_classification_data.drop("Atmospheric Pressure", axis=1)

# 4. Select the first 100 rows
subset_weather_data = new_weather_classification_data.iloc[:100, :]

# 5. Separate features (X) and target (y)
# Features: All columns except 'Weather Type'
weather_classification_data_x = subset_weather_data.drop("Weather Type", axis=1)
# Target: Only the 'Weather Type' column
weather_classification_data_y = subset_weather_data["Weather Type"]
# 6. Verify the shapes
print("Features (X) shape:", weather_classification_data_x.shape)
print("Target (y) shape:", weather_classification_data_y.shape)
print("Feature columns:", weather_classification_data_x.columns.tolist())