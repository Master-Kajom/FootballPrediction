import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers
import warnings

warnings.filterwarnings("ignore")

# Dummy data for the last 10 matches (1=Win, 0.5=Draw, 0=Loss)
man_utd_form = [1, 0, 1, 1, 0, 0.5, 1, 0, 1, 0.5]  # Last 10 matches
arsenal_form = [1, 1, 0, 1, 0.5, 1, 0, 1, 0.5, 1]  # Last 10 matches

# Dummy average goals scored/conceded in last 5 home/away matches
man_utd_home_goals = [2.1, 1.8, 2.3, 1.5, 2.0]  # Last 5 home matches
arsenal_away_goals = [1.8, 2.0, 1.5, 1.2, 1.7]  # Last 5 away matches
man_utd_home_conceded = [0.8, 1.2, 1.0, 0.5, 1.0]  # Last 5 home matches
arsenal_away_conceded = [0.9, 0.7, 1.1, 1.3, 0.8]  # Last 5 away matches

# Calculate form metrics
man_utd_form_avg = np.mean(man_utd_form)
arsenal_form_avg = np.mean(arsenal_form)
man_utd_attack_strength = np.mean(man_utd_home_goals) / np.mean(man_utd_home_goals + arsenal_away_goals)
arsenal_attack_strength = np.mean(arsenal_away_goals) / np.mean(man_utd_home_goals + arsenal_away_goals)
man_utd_defense_strength = np.mean(man_utd_home_conceded) / np.mean(man_utd_home_conceded + arsenal_away_conceded)
arsenal_defense_strength = np.mean(arsenal_away_conceded) / np.mean(man_utd_home_conceded + arsenal_away_conceded)

# Create features and target
X = np.array([
    [man_utd_form_avg, arsenal_form_avg, man_utd_attack_strength, 
     arsenal_attack_strength, man_utd_defense_strength, arsenal_defense_strength]
])

# Scale features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Load pre-trained model (in a real scenario, this would be pre-trained on historical data)
def create_model():
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(6,)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(3, activation='softmax')  # 3 outputs: Home Win, Draw, Away Win
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Create and compile model
model = create_model()

# Dummy weights (in a real scenario, these would be trained on historical data)
weights = model.get_weights()
for i in range(len(weights)):
    weights[i] = np.random.normal(0, 0.1, weights[i].shape)
model.set_weights(weights)

# Make prediction
prediction = model.predict(X_scaled)

# Calculate goal probabilities (Poisson distribution simplified)
man_utd_expected_goals = (man_utd_attack_strength * arsenal_defense_strength) * 1.8
arsenal_expected_goals = (arsenal_attack_strength * man_utd_defense_strength) * 1.5

# Normalize to get probabilities
man_utd_goal_prob = man_utd_expected_goals / (man_utd_expected_goals + arsenal_expected_goals)
arsenal_goal_prob = arsenal_expected_goals / (man_utd_expected_goals + arsenal_expected_goals)

# Print results
print("\n--- Match Prediction: Manchester United vs Arsenal ---")
print("\nTeam Statistics:")
print(f"Man Utd (Home) - Form: {man_utd_form_avg*100:.1f}%, "
      f"Avg Goals Scored: {np.mean(man_utd_home_goals):.2f}, "
      f"Avg Goals Conceded: {np.mean(man_utd_home_conceded):.2f}")
print(f"Arsenal (Away) - Form: {arsenal_form_avg*100:.1f}%, "
      f"Avg Goals Scored: {np.mean(arsenal_away_goals):.2f}, "
      f"Avg Goals Conceded: {np.mean(arsenal_away_conceded):.2f}")

print("\nGoal Scoring Probabilities:")
print(f"Man Utd chance of scoring: {man_utd_goal_prob*100:.1f}%")
print(f"Arsenal chance of scoring: {arsenal_goal_prob*100:.1f}%")

print("\nMatch Outcome Probabilities:")
print(f"Man Utd Win: {prediction[0][0]*100:.1f}%")
print(f"Draw: {prediction[0][1]*100:.1f}%")
print(f"Arsenal Win: {prediction[0][2]*100:.1f}%")

# Expected goals
print(f"\nExpected Goals (xG):")
print(f"Man Utd xG: {man_utd_expected_goals:.2f}")
print(f"Arsenal xG: {arsenal_expected_goals:.2f}")