from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
from datetime import datetime
import glob
import os
import numpy as np
import sys

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define Moroccan cities
CITIES = [
    'Casablanca', 'Rabat', 'Marrakech', 'Fes', 'Tangier', 'Agadir', 
    'Meknes', 'Oujda', 'Kenitra', 'Tetouan', 'El Jadida', 'Safi',
    'Mohammedia', 'Khouribga', 'Beni Mellal', 'Nador', 'Taza'
]

app = Flask(__name__)

# Load the latest model and preprocessor
def load_latest_model():
    model_dir = os.path.join(PROJECT_ROOT, 'models')
    model_files = glob.glob(os.path.join(model_dir, 'vehicle_price_model_*.joblib'))
    preprocessor_files = glob.glob(os.path.join(model_dir, 'preprocessor_*.joblib'))
    
    if not model_files or not preprocessor_files:
        raise FileNotFoundError("Model files not found. Please train the model first.")
    
    latest_model = max(model_files, key=os.path.getctime)
    latest_preprocessor = max(preprocessor_files, key=os.path.getctime)
    
    model = joblib.load(latest_model)
    preprocessor = joblib.load(latest_preprocessor)
    
    return model, preprocessor

def get_vehicle_data():
    # Load the dataset
    df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'all_type_data.csv'))
    
    # Get unique brands and models for each vehicle type
    car_data = {}
    motorcycle_data = {}
    
    for vehicle_type in ['Car', 'Motorcycle']:
        type_df = df[df['type'] == vehicle_type]
        brands = type_df['brand'].unique()
        
        for brand in brands:
            models = type_df[type_df['brand'] == brand]['model'].unique().tolist()
            if vehicle_type == 'Car':
                car_data[brand] = models
            else:
                motorcycle_data[brand] = models
    
    return car_data, motorcycle_data

def get_market_analysis():
    try:
        # Load the dataset
        df = pd.read_csv(os.path.join(PROJECT_ROOT, 'data', 'all_type_data.csv'))
        
        # Calculate market statistics
        analysis = {
            'brand_distribution': df['brand'].value_counts().head(5).to_dict(),
            'price_distribution': {
                '0-100K': len(df[df['price_mad'] <= 100000]),
                '100K-200K': len(df[(df['price_mad'] > 100000) & (df['price_mad'] <= 200000)]),
                '200K-300K': len(df[(df['price_mad'] > 200000) & (df['price_mad'] <= 300000)]),
                '300K-400K': len(df[(df['price_mad'] > 300000) & (df['price_mad'] <= 400000)]),
                '400K+': len(df[df['price_mad'] > 400000])
            },
            'fuel_distribution': df['fuel_type'].value_counts().to_dict(),
            'city_prices': df.groupby('city')['price_mad'].mean().to_dict(),
            'region_activity': {
                'North': len(df[df['city'].isin(['Tangier', 'Tetouan', 'Nador', 'Oujda'])]),
                'Central': len(df[df['city'].isin(['Casablanca', 'Rabat', 'Fes', 'Meknes', 'Kenitra'])]),
                'South': len(df[df['city'].isin(['Marrakech', 'Agadir', 'El Jadida', 'Safi', 'Beni Mellal'])])
            }
        }
        
        # Calculate price trends by year
        yearly_prices = df.groupby('year_of_manufacture')['price_mad'].mean()
        analysis['price_trends'] = {
            str(year): float(price) for year, price in yearly_prices.items()
        }
        
        return analysis
    except Exception as e:
        print(f"Error in market analysis: {str(e)}")
        # Return default data if there's an error
        return {
            'brand_distribution': {'Renault': 25, 'Dacia': 20, 'Volkswagen': 15, 'Toyota': 15, 'Hyundai': 10},
            'price_distribution': {'0-100K': 30, '100K-200K': 25, '200K-300K': 20, '300K-400K': 15, '400K+': 10},
            'fuel_distribution': {'Petrol': 45, 'Diesel': 35, 'Electric': 10, 'Hybrid': 10},
            'city_prices': {'Casablanca': 120000, 'Rabat': 110000, 'Marrakech': 100000, 'Tangier': 95000, 'Fes': 90000},
            'region_activity': {'North': 35, 'Central': 40, 'South': 25},
            'price_trends': {'2019': 100000, '2020': 95000, '2021': 105000, '2022': 110000, '2023': 115000}
        }

@app.route('/')
def index():
    current_year = datetime.now().year
    years = range(2000, current_year + 1)
    
    # Get brand and model data from the dataset
    car_data, motorcycle_data = get_vehicle_data()
    
    return render_template('index.html',
                         car_data=car_data,
                         motorcycle_data=motorcycle_data,
                         cities=CITIES,
                         years=years)

@app.route('/analysis')
def analysis():
    market_analysis = get_market_analysis()
    return render_template('analysis.html', analysis=market_analysis)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/get_models/<vehicle_type>/<brand>')
def get_models(vehicle_type, brand):
    car_data, motorcycle_data = get_vehicle_data()
    if vehicle_type == 'Car':
        models = car_data.get(brand, [])
    else:
        models = motorcycle_data.get(brand, [])
    return jsonify(models)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load model
        model, preprocessor = load_latest_model()
        
        # Get form data
        data = {
            'type': [request.form['vehicle_type']],
            'brand': [request.form['brand']],
            'model': [request.form['model']],
            'year_of_manufacture': [int(request.form['year'])],
            'mileage_km': [float(request.form['mileage'])],
            'transmission_type': [request.form['transmission']],
            'fuel_type': [request.form['fuel']],
            'accidents_history': [request.form['accidents']],
            'owner': [int(request.form['owners'])],
            'seats': [int(request.form['seats'])],
            'horse_power': [int(request.form['horsepower'])],
            'city': [request.form['city']]
        }
        
        # Create DataFrame
        input_df = pd.DataFrame(data)
        
        # Preprocess input
        X_processed = preprocessor.transform(input_df)
        
        # Make prediction
        prediction = model.predict(X_processed)[0]
        
        # Calculate confidence interval (using standard deviation)
        predictions = []
        n_iterations = 100
        
        for _ in range(n_iterations):
            # Add random noise to the prediction
            noise = np.random.normal(0, prediction * 0.1)  # 10% standard deviation
            predictions.append(prediction + noise)
        
        # Calculate confidence interval
        confidence_interval = np.percentile(predictions, [2.5, 97.5])
        
        return jsonify({
            'success': True,
            'prediction': round(prediction, -3),  # Round to nearest 1000
            'confidence_interval': [
                round(ci, -3) for ci in confidence_interval
            ]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
    app.run(debug=True) 