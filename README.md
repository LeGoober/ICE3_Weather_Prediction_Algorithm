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



Notes

The dataset (weather_classification_data.csv) contains features like Temperature, Humidity, and Weather Type, used to train the Decision Tree.
Ensure .env (if used) is added to .gitignore for sensitive data.
Outputs (e.g., Decision Tree plots, confusion matrices) are saved in output/figures/.

Contact
For questions, contact the team via the repository's issue tracker.