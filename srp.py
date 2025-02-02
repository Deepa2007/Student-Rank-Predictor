import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Fetch data from API endpoints
def fetch_data(url):
    response = requests.get(url)
    return response.json()

current_quiz_data = fetch_data('https://api.example.com/current-quiz')
historical_quiz_data = fetch_data('https://api.example.com/historical-quiz')

# Convert to DataFrame
current_df = pd.DataFrame(current_quiz_data)
historical_df = pd.DataFrame(historical_quiz_data)

# Data preprocessing and feature engineering
# (Example: Calculate average score, topic-wise accuracy, etc.)
# ...

# Split data into features and target
X = historical_df[['feature1', 'feature2', 'feature3']]
y = historical_df['neet_rank']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Predict rank for a new student
new_student_data = pd.DataFrame([{'feature1': value1, 'feature2': value2, 'feature3': value3}])
predicted_rank = model.predict(new_student_data)
print(f'Predicted NEET Rank: {predicted_rank[0]}')

# Bonus: Predict college
def predict_college(predicted_rank):
    # Logic to map rank to college based on cutoff data
    # ...
    return likely_college

likely_college = predict_college(predicted_rank)
print(f'Likely College: {likely_college}')
