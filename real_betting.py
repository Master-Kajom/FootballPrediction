import os
import csv
import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
from typing import Dict, Tuple, Optional
import math

# ML and Data Processing
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# TensorFlow/Keras
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns

# Local imports
from config import (
    DEFAULT_COMPETITION, CURRENT_SEASON,
    normalize_team_name, safe_request, API_KEY
)
from fetch_match_stats import get_match_stats, analyze_matches
from fetch_head_to_head import get_head_to_head
from goal_difference import get_teams_goal_difference

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Constants
COMPETITION = DEFAULT_COMPETITION
SEASON = CURRENT_SEASON

# Model Configuration
MODEL_CHECKPOINT = 'best_model.weights.h5'  # Changed to .h5 extension for weights
PREDICTION_HISTORY = 'prediction_history.csv'

# Feature Engineering Functions

def get_team_form_data(home_team: str, away_team: str, competition: str = COMPETITION) -> Tuple[Dict, Dict]:
    """Get form data for both teams"""
    print(f"\nFetching form data for {home_team} vs {away_team}...")
    
    # Get match stats for both teams
    print(f"\n=== DEBUG: Getting match stats for {home_team} and {away_team} ===")
    home_stats, away_stats = get_match_stats(home_team, away_team, competition)
    
    #print(f"\n=== DEBUG: Raw home_stats ===\n{home_stats}")
    #print(f"\n=== DEBUG: Raw away_stats ===\n{away_stats}")
    
    if not home_stats or not away_stats:
        raise ValueError("Could not fetch match statistics for one or both teams")
    
    # Calculate form metrics (last 5 matches)
    home_form = {
        'goals_scored': home_stats['goals_scored'],
        'goals_conceded': home_stats['goals_conceded'],
        'results': home_stats['results'],
        'avg_goals_scored': np.mean(home_stats['goals_scored']) if home_stats['goals_scored'] else 0,
        'avg_goals_conceded': np.mean(home_stats['goals_conceded']) if home_stats['goals_conceded'] else 0,
        'form_avg': np.mean(home_stats['results']) if home_stats['results'] else 0.5
    }
    
    away_form = {
        'goals_scored': away_stats['goals_scored'],
        'goals_conceded': away_stats['goals_conceded'],
        'results': away_stats['results'],
        'avg_goals_scored': np.mean(away_stats['goals_scored']) if away_stats['goals_scored'] else 0,
        'avg_goals_conceded': np.mean(away_stats['goals_conceded']) if away_stats['goals_conceded'] else 0,
        'form_avg': np.mean(away_stats['results']) if away_stats['results'] else 0.5
    }
    
    #print(f"\n=== DEBUG: Processed home_form ===\n{home_form}")
    #print(f"\n=== DEBUG: Processed away_form ===\n{away_form}")
    
    return home_form, away_form

def get_head_to_head_data(home_team: str, away_team: str) -> Dict:
    """Get head-to-head statistics"""
    print(f"\nFetching head-to-head data for {home_team} vs {away_team}...")
    h2h_stats = get_head_to_head(home_team, away_team)
    
    print(f"\n=== DEBUG: Raw h2h_stats ===\n{h2h_stats}")
    
    if not h2h_stats:
        print("Warning: Could not fetch head-to-head data. Using default values.")
        home_team_key = home_team.lower().replace(' ', '_').replace('_fc', '').replace('_afc', '')
        away_team_key = away_team.lower().replace(' ', '_').replace('_fc', '').replace('_afc', '')
        
        # Default values if no H2H data
        default_stats = {
            f"{home_team_key}_home_h2h_avg": 0.5,
            f"{away_team_key}_away_h2h_avg": 0.5
        }
        print(f"Using default h2h stats: {default_stats}")
        return default_stats
    
    return h2h_stats

def get_goal_difference_data(home_team: str, away_team: str, 
                           season: int = SEASON, 
                           competition: str = COMPETITION) -> Tuple[Dict, Dict]:
    """Get goal difference data for both teams"""
    print(f"\nFetching goal difference data for {season-1}-{season} season...")
    
    try:
        home_gd, away_gd = get_teams_goal_difference(
            competition, season, home_team, away_team
        )
        
        #print(f"\n=== DEBUG: Raw home_gd ===\n{home_gd}")
        #print(f"\n=== DEBUG: Raw away_gd ===\n{away_gd}")
        
        # Normalize goal difference to 0-1 range
        max_gd = max(
            abs(home_gd.get('goalDifference', 0)), 
            abs(away_gd.get('goalDifference', 0)), 
            1  # Avoid division by zero
        )
        
        home_gd_normalized = (home_gd.get('goalDifference', 0) + max_gd) / (2 * max_gd)
        away_gd_normalized = (away_gd.get('goalDifference', 0) + max_gd) / (2 * max_gd)
        
        home_result = {
            'goal_difference': home_gd.get('goalDifference', 0),
            'goals_for': home_gd.get('goalsFor', 0),
            'goals_against': home_gd.get('goalsAgainst', 0),
            'matches_played': home_gd.get('matchesPlayed', 0),
            'normalized': home_gd_normalized
        }
        
        away_result = {
            'goal_difference': away_gd.get('goalDifference', 0),
            'goals_for': away_gd.get('goalsFor', 0),
            'goals_against': away_gd.get('goalsAgainst', 0),
            'matches_played': away_gd.get('matchesPlayed', 0),
            'normalized': away_gd_normalized
        }
        
        print(f"\n=== DEBUG: Processed home_gd ===\n{home_result}")
        print(f"\n=== DEBUG: Processed away_gd ===\n{away_result}")
        
        return home_result, away_result
        
    except Exception as e:
        print(f"Error fetching goal difference data: {e}")
        # Return neutral values if there's an error
        neutral = {
            'goal_difference': 0,
            'goals_for': 0,
            'goals_against': 0,
            'matches_played': 0,
            'normalized': 0.5
        }
        print(f"Using neutral values due to error: {neutral}")
        return neutral, neutral

def prepare_features(home_form: Dict, away_form: Dict, h2h_stats: Dict, 
                   home_gd: Dict, away_gd: Dict) -> np.ndarray:
    """Prepare feature vector for the model"""
    print("\n=== DEBUG: Preparing features ===")
    # Extract team names from h2h_stats keys if available
    home_team_key = next((k for k in h2h_stats.keys() if 'home_h2h_avg' in k), '').replace('_home_h2h_avg', '')
    away_team_key = next((k for k in h2h_stats.keys() if 'away_h2h_avg' in k), '').replace('_away_h2h_avg', '')
    
    # Get H2H stats with fallback to 0.5 (neutral) if not found
    home_h2h = h2h_stats.get(f'{home_team_key}_home_h2h_avg', 0.5)
    away_h2h = h2h_stats.get(f'{away_team_key}_away_h2h_avg', 0.5)
    
    print(f"Home form: {home_form['form_avg']:.2f}")
    print(f"Away form: {away_form['form_avg']:.2f}")
    print(f"Home H2H: {home_h2h:.2f}")
    print(f"Away H2H: {away_h2h:.2f}")
    print(f"Home GD: {home_gd['normalized']:.2f}")
    print(f"Away GD: {away_gd['normalized']:.2f}")

    # Extract and scale features
    features = [
        home_form['form_avg'],                      # Home team form (0-1)
        away_form['form_avg'],                      # Away team form (0-1)
        home_form['avg_goals_scored'] / 5.0,        # Normalize by max expected goals
        away_form['avg_goals_scored'] / 5.0,        # Normalize by max expected goals
        home_form['avg_goals_conceded'] / 5.0,      # Normalize by max expected goals
        away_form['avg_goals_conceded'] / 5.0,      # Normalize by max expected goals
        home_h2h,                                   # Home team H2H performance at home (0-1)
        away_h2h,                                   # Away team H2H performance away (0-1)
        home_gd['normalized'],                      # Home team goal difference (normalized)
        away_gd['normalized']                       # Away team goal difference (normalized)
    ]
    
    return np.array([features])  # Return as 2D array for the model

# Model Functions

def create_model(input_shape=10):
    """Create and compile the neural network model"""
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.3),
        layers.BatchNormalization(),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.BatchNormalization(),
        layers.Dense(3, activation='softmax')  # 3 outputs: home win, draw, away win
    ])
    
    # Custom optimizer with learning rate schedule
    initial_learning_rate = 0.001
    lr_schedule = keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate, decay_steps=1000, decay_rate=0.9, staircase=True)
    
    optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
    
    model.compile(optimizer=optimizer,
                 loss='categorical_crossentropy',
                 metrics=['accuracy', 
                         keras.metrics.AUC(name='auc'),
                         keras.metrics.Precision(name='precision'),
                         keras.metrics.Recall(name='recall')])
    
    return model

def load_or_train_model(model_path=MODEL_CHECKPOINT):
    """Load a trained model or create a new one if none exists"""
    if os.path.exists(model_path):
        try:
            print("Loading pre-trained model...")
            model = create_model()
            model.load_weights(model_path)
            print("Model loaded successfully!")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Creating a new model...")
    
    # If we get here, either the model doesn't exist or there was an error loading it
    print("Training a new model with sample data...")
    
    # Generate some sample data for initial training
    X = np.random.rand(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 3, size=100)  # 3 classes
    
    # Convert to one-hot encoding
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=3)
    
    # Train the model
    model = create_model()
    model.fit(X, y_one_hot, epochs=10, batch_size=16, verbose=1)
    
    # Save the model
    model.save_weights(model_path)
    print(f"Model saved to {model_path}")
    
    return model

def predict_match(home_team: str, away_team: str, competition: str = COMPETITION, 
                 season: int = SEASON) -> Optional[Dict]:
    """Predict match outcome using all available data"""
    try:
        print(f"\n=== Starting prediction for {home_team} vs {away_team} ===")
        
        # Get all required data
        print("Fetching team form data...")
        home_form, away_form = get_team_form_data(home_team, away_team, competition)
        
        print("Fetching head-to-head data...")
        h2h_stats = get_head_to_head_data(home_team, away_team)
        
        print("Fetching goal difference data...")
        home_gd, away_gd = get_goal_difference_data(home_team, away_team, season, competition)
        
        # Prepare features
        print("Preparing features...")
        X = prepare_features(home_form, away_form, h2h_stats, home_gd, away_gd)
        
        # Load or train model
        model = load_or_train_model()
        
        # Make prediction
        print("Making prediction...")
        prediction = model.predict(X, verbose=0)[0]
        
        # Calculate expected goals (improved calculation)
        LEAGUE_AVG_GOALS = 1.5
        HOME_ADVANTAGE = 0.3
        
        # Calculate team strengths relative to league average
        home_attack = (home_form.get('avg_goals_scored', 1.0) / max(LEAGUE_AVG_GOALS, 0.1))  # Avoid division by zero
        home_defense = (home_form.get('avg_goals_conceded', 1.0) / max(LEAGUE_AVG_GOALS, 0.1))
        away_attack = (away_form.get('avg_goals_scored', 1.0) / max(LEAGUE_AVG_GOALS, 0.1))
        away_defense = (away_form.get('avg_goals_conceded', 1.0) / max(LEAGUE_AVG_GOALS, 0.1))
        
        # Base xG calculation with team strengths and home advantage
        home_xg = (home_attack * away_defense * LEAGUE_AVG_GOALS) + HOME_ADVANTAGE
        away_xg = (away_attack * home_defense * LEAGUE_AVG_GOALS) - (HOME_ADVANTAGE * 0.5)
        
        # Apply form influence (recent 5 matches)
        form_influence = 0.25  # Increased influence of recent form
        home_xg *= (1 + (home_form.get('form_avg', 0.5) - 0.5) * form_influence)
        away_xg *= (1 + (away_form.get('form_avg', 0.5) - 0.5) * form_influence)
        
        # Apply H2H influence (up to 30% adjustment based on H2H performance)
        h2h_influence = 0.3  # Increased H2H influence
        home_xg *= (1 + (h2h_stats.get(f'{home_team.lower().replace(" ", "_").replace("_fc", "").replace("_afc", "")}_home_h2h_avg', 0.5) - 0.5) * h2h_influence)
        away_xg *= (1 + (h2h_stats.get(f'{away_team.lower().replace(" ", "_").replace("_fc", "").replace("_afc", "")}_away_h2h_avg', 0.5) - 0.5) * h2h_influence)
        
        # Ensure xG stays within reasonable bounds
        home_xg = max(0.1, min(4.0, home_xg))
        away_xg = max(0.1, min(4.0, away_xg))
        
        # Calculate win probabilities based on xG using Poisson distribution
        def poisson_prob(goals, expected):
            return (expected ** goals) * (2.71828 ** -expected) / math.factorial(goals)
        
        # Calculate match outcome probabilities
        home_win_prob = 0
        draw_prob = 0
        away_win_prob = 0
        
        for i in range(0, 5):  # 0-4 goals for home team
            for j in range(0, 5):  # 0-4 goals for away team
                prob = poisson_prob(i, home_xg) * poisson_prob(j, away_xg)
                if i > j:
                    home_win_prob += prob
                elif i == j:
                    draw_prob += prob
                else:
                    away_win_prob += prob
        
        # Normalize probabilities to sum to 1
        total = home_win_prob + draw_prob + away_win_prob
        if total > 0:
            home_win_prob /= total
            draw_prob /= total
            away_win_prob /= total
        
        # If model prediction is available, blend it with xG-based prediction
        if prediction is not None:
            xg_weight = 0.6  # Weight for xG-based prediction
            model_weight = 0.4  # Weight for model prediction
            
            home_win_prob = (home_win_prob * xg_weight) + (prediction[0] * model_weight)
            draw_prob = (draw_prob * xg_weight) + (prediction[1] * model_weight)
            away_win_prob = (away_win_prob * xg_weight) + (prediction[2] * model_weight)
            
            # Renormalize
            total = home_win_prob + draw_prob + away_win_prob
            if total > 0:
                home_win_prob /= total
                draw_prob /= total
                away_win_prob /= total
        
        return {
            'home_win_prob': float(home_win_prob),
            'draw_prob': float(draw_prob),
            'away_win_prob': float(away_win_prob),
            'home_expected_goals': round(home_xg, 2),
            'away_expected_goals': round(away_xg, 2),
            'home_form_avg': round(home_form.get('form_avg', 0.5), 2),
            'away_form_avg': round(away_form.get('form_avg', 0.5), 2),
            'home_goal_difference': home_gd.get('goal_difference', 0),
            'away_goal_difference': away_gd.get('goal_difference', 0)
        }
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        import traceback
        traceback.print_exc()
        return None

def print_prediction(home_team: str, away_team: str, prediction: Dict):
    """Print prediction results in a readable format"""
    if not prediction:
        print("Could not generate prediction due to errors.")
        return
    
    print("\n" + "="*50)
    print(f"MATCH PREDICTION: {home_team} vs {away_team}")
    print("="*50)
    
    print(f"\n{'Outcome':<15} {'Probability':>12}")
    print("-"*30)
    print(f"{home_team} Win: {prediction['home_win_prob']*100:>10.1f}%")
    print(f"Draw: {prediction['draw_prob']*100:>18.1f}%")
    print(f"{away_team} Win: {prediction['away_win_prob']*100:>10.1f}%")
    
    print(f"\n{'Expected Goals (xG):':<25}")
    print("-"*30)
    print(f"{home_team}: {prediction['home_expected_goals']:>5.2f}")
    print(f"{away_team}: {prediction['away_expected_goals']:>5.2f}")
    
    print(f"\n{'Team Form (Last 5):':<25}")
    print("-"*30)
    print(f"{home_team}: {prediction['home_form_avg']*100:>5.1f}%")
    print(f"{away_team}: {prediction['away_form_avg']*100:>5.1f}%")
    
    print(f"\n{'Goal Difference:':<25}")
    print("-"*30)
    print(f"{home_team}: {prediction['home_goal_difference']:>+3d}")
    print(f"{away_team}: {prediction['away_goal_difference']:>+3d}")
    print("\n" + "="*50 + "\n")

def save_prediction_to_history(home_team: str, away_team: str, prediction: Dict, 
                             competition: str, season: int):
    """Save prediction to history file for future reference"""
    try:
        header = not os.path.exists(PREDICTION_HISTORY)
        
        with open(PREDICTION_HISTORY, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if header:
                writer.writerow([
                    'timestamp', 'home_team', 'away_team', 'competition', 'season',
                    'home_win_prob', 'draw_prob', 'away_win_prob',
                    'home_expected_goals', 'away_expected_goals',
                    'home_form_avg', 'away_form_avg'
                ])
            
            writer.writerow([
                datetime.now().isoformat(),
                home_team, away_team, competition, f"{season-1}-{season%100}",
                prediction['home_win_prob'], prediction['draw_prob'], prediction['away_win_prob'],
                prediction['home_expected_goals'], prediction['away_expected_goals'],
                prediction['home_form_avg'], prediction['away_form_avg']
            ])
            
    except Exception as e:
        print(f"Warning: Could not save prediction to history: {e}")

def show_similar_matches(home_team: str, away_team: str, competition: str, limit: int = 5):
    """Show similar historical matches for reference"""
    try:
        # In a real implementation, this would query a database of historical matches
        # For now, we'll just show a message
        print("\n[Similar historical matches would be shown here]")
        print(f"Searching for similar {competition} matches between {home_team} and {away_team}...")
        
    except Exception as e:
        print(f"\nCould not retrieve similar matches: {e}")

def main():
    """Main function to run the football match prediction system"""
    print("\n" + "="*60)
    print("FOOTBALL MATCH PREDICTION SYSTEM".center(60))
    print("="*60 + "\n")
    
    # Get user input for match prediction
    while True:
        try:
            print("\nEnter the teams for prediction (or 'q' to quit):")
            home_team = input("Home Team: ").strip()
            if home_team.lower() == 'q':
                break
                
            away_team = input("Away Team: ").strip()
            if away_team.lower() == 'q':
                break
                
            competition = input(f"Competition (default: {COMPETITION}): ").strip() or COMPETITION
            season_input = input(f"Season (default: {SEASON-1}-{SEASON%100}): ").strip()
            
            if season_input:
                try:
                    season = int(season_input.split('-')[0]) + 1  # Convert to end year format
                except (ValueError, IndexError):
                    print("Invalid season format. Using default season.")
                    season = SEASON
            else:
                season = SEASON
            
            print(f"\nAnalyzing {home_team} vs {away_team}...")
            
            # Get prediction
            prediction = predict_match(home_team, away_team, competition, season)
            
            if prediction:
                # Print prediction results
                print_prediction(home_team, away_team, prediction)
                
                # Save prediction to history
                save_prediction_to_history(home_team, away_team, prediction, competition, season)
                
                # Show similar historical matches
                show_similar_matches(home_team, away_team, competition)
                
            else:
                print("Failed to generate prediction. Please check the team names and try again.")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            continue

if __name__ == "__main__":
    try:
        # Check API key
        api_key = API_KEY
        if not api_key:
            print("Error: FOOTBALL_DATA_API_KEY not found in environment variables or .env file.")
            print("Please set up your API key as described in the README.md")
            exit(1)
            
        main()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        print("Please check your internet connection and try again later.")
    finally:
        print("\nThank you for using the Football Match Prediction System!")
