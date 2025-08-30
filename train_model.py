import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import os

def create_model(input_shape=10):
    """Create a simple neural network model"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape,)),
        Dropout(0.3),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dropout(0.2),
        BatchNormalization(),
        Dense(3, activation='softmax')  # 3 outputs: home win, draw, away win
    ])
    
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def generate_sample_data(num_samples=1000):
    """Generate sample training data"""
    # Generate random features (10 features as expected by the model)
    X = np.random.rand(num_samples, 10)
    
    # Generate target variable (0=home win, 1=draw, 2=away win)
    # Make home wins slightly more likely than away wins, with draws being least likely
    y = np.random.choice([0, 1, 2], size=num_samples, p=[0.45, 0.3, 0.25])
    
    return X, y

def train():
    # Create output directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Generate sample data
    print("Generating sample data...")
    X, y = generate_sample_data(1000)
    
    # Convert to one-hot encoding
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=3)
    
    # Create and train model
    print("Creating and training model...")
    model = create_model()
    
    # Simple training with validation split
    model.fit(
        X, y_one_hot,
        epochs=20,
        batch_size=32,
        validation_split=0.2,
        verbose=1
    )
    
    # Save the model
    model_path = 'best_model.weights.h5'  # Changed to .h5 extension for weights
    model.save_weights(model_path)
    print(f"\nModel weights saved to {model_path}")
    
    # Also save the full model
    full_model_path = 'best_model.keras'
    model.save(full_model_path)
    print(f"Full model saved to {full_model_path}")
    
    # Print sample prediction
    sample_input = np.random.rand(1, 10)
    prediction = model.predict(sample_input, verbose=0)
    print("\nSample prediction:")
    print(f"Home Win: {prediction[0][0]*100:.1f}%")
    print(f"Draw: {prediction[0][1]*100:.1f}%")
    print(f"Away Win: {prediction[0][2]*100:.1f}%")

if __name__ == "__main__":
    print("Starting model training...")
    train()
    print("\nTraining completed!")
