import joblib
import argparse
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('concrete_compressive_model.pkl')

# Define the feature names as seen during training
feature_names = [
    'Cement (component 1)(kg in a m^3 mixture)', 
    'Blast Furnace Slag (component 2)(kg in a m^3 mixture)',
    'Fly Ash (component 3)(kg in a m^3 mixture)', 
    'Water  (component 4)(kg in a m^3 mixture)', 
    'Superplasticizer (component 5)(kg in a m^3 mixture)', 
    'Coarse Aggregate  (component 6)(kg in a m^3 mixture)', 
    'Fine Aggregate (component 7)(kg in a m^3 mixture)', 
    'Age (day)'
]

# Function to predict compressive strength
def predict_compressive_strength(input_data):
    # Convert input data to a pandas DataFrame with correct column names
    input_df = pd.DataFrame([input_data], columns=feature_names)
    
    # Make the prediction using the model
    prediction = model.predict(input_df)
    
    return prediction[0]

if __name__ == '__main__':
    # Command line argument parser
    parser = argparse.ArgumentParser(description='Predict Concrete Compressive Strength')
    
    # Add the input parameters (same ones used for training)
    parser.add_argument('--cement', type=float, required=True, help='Cement content')
    parser.add_argument('--slag', type=float, required=True, help='Blast Furnace Slag content')
    parser.add_argument('--ash', type=float, required=True, help='Fly Ash content')
    parser.add_argument('--water', type=float, required=True, help='Water content')
    parser.add_argument('--superplasticizer', type=float, required=True, help='Superplasticizer content')
    parser.add_argument('--coarse_aggregate', type=float, required=True, help='Coarse Aggregate content')
    parser.add_argument('--fine_aggregate', type=float, required=True, help='Fine Aggregate content')
    parser.add_argument('--age', type=float, required=True, help='Age of the concrete')

    # Parse the command line arguments
    args = parser.parse_args()
    
    # Create input data from parsed arguments
    input_data = [args.cement, args.slag, args.ash, args.water, args.superplasticizer, 
                  args.coarse_aggregate, args.fine_aggregate, args.age]
    
    # Get the prediction from the model
    result = predict_compressive_strength(input_data)
    
    # Print the predicted compressive strength
    print(f"Predicted Compressive Strength: {result:.2f} MPa")
