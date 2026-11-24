import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
from datetime import datetime
import os

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class VehiclePriceModel:
    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.feature_names = None
        
    def preprocess_data(self, df):
        """Create preprocessing pipeline"""
        numeric_features = ['year_of_manufacture', 'mileage_km', 'owner', 'seats', 'horse_power']
        categorical_features = ['type', 'brand', 'model', 'transmission_type', 'fuel_type', 
                              'accidents_history', 'city']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first', sparse_output=False))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        self.feature_names = numeric_features + categorical_features
        
    def train(self, df, validation_split=True):
        """Train the model with various validation strategies"""
        # Prepare data
        X = df[self.feature_names]
        y = df['price_mad']
        
        # Initialize model
        self.model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        if validation_split:
            # First split: train+validation and test
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                X, y, test_size=0.15, random_state=42
            )
            
            # Second split: train and validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train_val, y_train_val, test_size=0.15/0.85, random_state=42
            )
            
            # Fit preprocessor on training data
            X_train_processed = self.preprocessor.fit_transform(X_train)
            X_val_processed = self.preprocessor.transform(X_val)
            X_test_processed = self.preprocessor.transform(X_test)
            
            # Train model
            self.model.fit(X_train_processed, y_train)
            
            # Evaluate on validation set
            val_predictions = self.model.predict(X_val_processed)
            val_mae = mean_absolute_error(y_val, val_predictions)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_predictions))
            val_r2 = r2_score(y_val, val_predictions)
            
            # Evaluate on test set
            test_predictions = self.model.predict(X_test_processed)
            test_mae = mean_absolute_error(y_test, test_predictions)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
            test_r2 = r2_score(y_test, test_predictions)
            
            print("\nValidation Set Metrics:")
            print(f"MAE: {val_mae:,.0f} MAD")
            print(f"RMSE: {val_rmse:,.0f} MAD")
            print(f"R²: {val_r2:.3f}")
            
            print("\nTest Set Metrics:")
            print(f"MAE: {test_mae:,.0f} MAD")
            print(f"RMSE: {test_rmse:,.0f} MAD")
            print(f"R²: {test_r2:.3f}")
        
        # K-fold cross-validation
        print("\nPerforming 5-fold cross-validation...")
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
            X_train_fold = X.iloc[train_idx]
            y_train_fold = y.iloc[train_idx]
            X_val_fold = X.iloc[val_idx]
            y_val_fold = y.iloc[val_idx]
            
            # Preprocess fold data
            X_train_processed = self.preprocessor.fit_transform(X_train_fold)
            X_val_processed = self.preprocessor.transform(X_val_fold)
            
            # Train and evaluate
            self.model.fit(X_train_processed, y_train_fold)
            predictions = self.model.predict(X_val_processed)
            
            fold_mae = mean_absolute_error(y_val_fold, predictions)
            fold_rmse = np.sqrt(mean_squared_error(y_val_fold, predictions))
            fold_r2 = r2_score(y_val_fold, predictions)
            
            cv_scores.append({
                'mae': fold_mae,
                'rmse': fold_rmse,
                'r2': fold_r2
            })
            
            print(f"\nFold {fold} Metrics:")
            print(f"MAE: {fold_mae:,.0f} MAD")
            print(f"RMSE: {fold_rmse:,.0f} MAD")
            print(f"R²: {fold_r2:.3f}")
        
        # Calculate average CV scores
        avg_mae = np.mean([score['mae'] for score in cv_scores])
        avg_rmse = np.mean([score['rmse'] for score in cv_scores])
        avg_r2 = np.mean([score['r2'] for score in cv_scores])
        
        print("\nAverage Cross-Validation Metrics:")
        print(f"MAE: {avg_mae:,.0f} MAD")
        print(f"RMSE: {avg_rmse:,.0f} MAD")
        print(f"R²: {avg_r2:.3f}")
    
    def predict(self, input_data):
        """Make predictions with confidence intervals"""
        X_processed = self.preprocessor.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(X_processed)[0]
        
        # Calculate confidence interval (using standard deviation)
        predictions = []
        n_iterations = 100
        
        for _ in range(n_iterations):
            # Add random noise to the prediction
            noise = np.random.normal(0, prediction * 0.1)  # 10% standard deviation
            predictions.append(prediction + noise)
        
        # Calculate confidence interval
        confidence_interval = np.percentile(predictions, [2.5, 97.5])
        
        return {
            'prediction': round(prediction, -3),  # Round to nearest 1000
            'confidence_interval': [round(ci, -3) for ci in confidence_interval]
        }
    
    def save_model(self, model_path=None):
        """Save the trained model and preprocessor"""
        if model_path is None:
            model_path = os.path.join(PROJECT_ROOT, 'models')
            
        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = os.path.join(model_path, f"vehicle_price_model_{timestamp}.joblib")
        preprocessor_filename = os.path.join(model_path, f"preprocessor_{timestamp}.joblib")
        
        joblib.dump(self.model, model_filename)
        joblib.dump(self.preprocessor, preprocessor_filename)
        print(f"\nModel saved as {model_filename}")
        print(f"Preprocessor saved as {preprocessor_filename}")

if __name__ == "__main__":
    # Load data
    data_file = os.path.join(PROJECT_ROOT, 'data', 'all_type_data.csv')
    df = pd.read_csv(data_file)
    
    # Clean data
    df['accidents_history'] = df['accidents_history'].fillna('None')
    df['transmission_type'] = df['transmission_type'].fillna('Manual')
    df['fuel_type'] = df['fuel_type'].fillna('Petrol')
    df['owner'] = df['owner'].fillna(1)
    df['seats'] = df['seats'].fillna(df.groupby('type')['seats'].transform('median'))
    df['horse_power'] = df['horse_power'].fillna(df.groupby('type')['horse_power'].transform('median'))
    df['mileage_km'] = df['mileage_km'].fillna(df.groupby('type')['mileage_km'].transform('median'))
    df['year_of_manufacture'] = df['year_of_manufacture'].fillna(2020)
    
    # Add city data with realistic distribution
    cities = [
        'Casablanca', 'Rabat', 'Marrakech', 'Fes', 'Tangier', 'Agadir', 
        'Meknes', 'Oujda', 'Kenitra', 'Tetouan', 'El Jadida', 'Safi',
        'Mohammedia', 'Khouribga', 'Beni Mellal', 'Nador', 'Taza'
    ]
    
    # City weights based on population and market size
    city_weights = [
        0.25,    # Casablanca (economic capital)
        0.15,    # Rabat (administrative capital)
        0.12,    # Marrakech
        0.08,    # Fes
        0.08,    # Tangier
        0.06,    # Agadir
        0.04,    # Meknes
        0.04,    # Oujda
        0.03,    # Kenitra
        0.03,    # Tetouan
        0.02,    # El Jadida
        0.02,    # Safi
        0.02,    # Mohammedia
        0.02,    # Khouribga
        0.02,    # Beni Mellal
        0.01,    # Nador
        0.01     # Taza
    ]
    
    # Assign cities based on weighted distribution
    df['city'] = np.random.choice(cities, size=len(df), p=city_weights)
    
    # City factors (relative to Casablanca as baseline 1.0)
    CITY_FACTORS = {
        'Casablanca': 1.0,    # Economic capital, baseline
        'Rabat': 1.05,        # Administrative capital, slightly higher
        'Marrakech': 0.95,    # Tourist city
        'Fes': 0.90,         # Historical city
        'Tangier': 1.02,     # Port city, industrial
        'Agadir': 0.92,      # Tourist city
        'Meknes': 0.88,      # Historical city
        'Oujda': 0.85,       # Border city
        'Kenitra': 0.87,     # Industrial city
        'Tetouan': 0.86,     # Northern city
        'El Jadida': 0.89,   # Coastal city
        'Safi': 0.84,        # Port city
        'Mohammedia': 0.95,  # Industrial city
        'Khouribga': 0.83,   # Mining city
        'Beni Mellal': 0.82, # Agricultural city
        'Nador': 0.85,       # Port city
        'Taza': 0.80         # Small city
    }
    
    # Generate price_mad based on features
    def calculate_base_price(row):
        # Base prices by vehicle type and brand premium
        brand_premiums = {
            'Car': {
                'Ferrari': 8.0, 'Maserati': 7.0, 'Tesla': 6.0,
                'Mercedes-Benz': 5.0, 'BMW': 4.8, 'Audi': 4.5, 'Land Rover': 4.2,
                'Jaguar': 4.0, 'Volkswagen': 3.0, 'Toyota': 2.8, 'Nissan': 2.5,
                'Hyundai': 2.2, 'Kia': 2.0, 'Ford': 2.2, 'Peugeot': 2.0,
                'Renault': 1.8, 'Citroën': 1.8, 'Fiat': 1.6, 'Dacia': 1.4,
                'BYD': 2.5, 'MG': 1.8, 'Opel': 1.9, 'Seat': 2.0, 'Skoda': 2.1
            },
            'Motorcycle': {
                'Ducati': 5.0, 'BMW Motorrad': 4.5, 'Harley-Davidson': 4.2,
                'KTM': 4.0, 'Triumph': 3.8, 'Kawasaki': 3.5, 'Honda': 3.2,
                'Yamaha': 3.0, 'Suzuki': 2.8, 'Aprilia': 3.5, 'Royal Enfield': 2.0,
                'Zontes': 1.8, 'Kymco': 1.6, 'MBK': 1.5, 'Lifan': 1.2
            }
        }

        # Get brand premium factor
        brand_premium = brand_premiums[row['type']].get(row['brand'], 1.0)
        
        # Base price ranges with brand premium
        if row['type'] == 'Car':
            base_price = 120000 * brand_premium  # Base 120k MAD for standard car
        else:
            base_price = 25000 * brand_premium   # Base 25k MAD for standard motorcycle

        # Model-specific adjustments (example: premium models)
        premium_models = {
            'G-Class': 2.5, 'S-Class': 2.3, 'Range Rover': 2.2, 'Model S': 2.0,
            'A8': 2.0, '7 Series': 2.0, 'Panamera': 2.0, 'R8': 2.5,
            'Multistrada V4': 1.8, 'R1250GS': 1.7, 'Pan America 1250': 1.6,
            'CBR1000RR': 1.5, 'Ninja H2': 1.8, 'S1000RR': 1.7
        }
        model_premium = premium_models.get(row['model'], 1.0)
        base_price *= model_premium

        # Age depreciation (newer = more expensive)
        current_year = 2025
        age = current_year - row['year_of_manufacture']
        if age <= 0:
            year_factor = 1.1  # Premium for newer models
        else:
            year_factor = max(0.4, 1.0 - (age * 0.06))  # 6% depreciation per year, minimum 40% value

        # Mileage depreciation
        max_mileage = 300000 if row['type'] == 'Car' else 100000
        mileage_factor = max(0.4, 1.0 - (row['mileage_km'] / max_mileage))

        # Accident history
        accident_factors = {'None': 1.0, 'Minor': 0.85, 'Major': 0.65}
        accident_factor = accident_factors[row['accidents_history']]

        # Owner count depreciation
        owner_factor = max(0.7, 1.0 - ((row['owner'] - 1) * 0.12))  # 12% per owner, minimum 70%

        # Horsepower premium
        hp_reference = 400 if row['type'] == 'Car' else 150
        hp_factor = 1.0 + (row['horse_power'] / hp_reference) * 0.5

        # Transmission type
        transmission_factor = 1.1 if row['transmission_type'] == 'Automatic' else 1.0

        # Fuel type
        fuel_factors = {
            'Electric': 1.3,
            'Hybrid': 1.2,
            'Petrol': 1.0,
            'Diesel': 0.95
        }
        fuel_factor = fuel_factors.get(row['fuel_type'], 1.0)

        # City factor
        city_factor = CITY_FACTORS.get(row['city'], 1.0)

        # Calculate final price
        price = (base_price * year_factor * mileage_factor * accident_factor * 
                owner_factor * hp_factor * transmission_factor * fuel_factor * city_factor)

        # Add random variation (±5%)
        variation = np.random.uniform(0.95, 1.05)
        
        # Ensure minimum prices
        min_price = 50000 if row['type'] == 'Car' else 8000
        return max(round(price * variation, -3), min_price)

    # Generate prices
    df['price_mad'] = df.apply(calculate_base_price, axis=1)
    
    # Initialize and train model with better parameters
    model = VehiclePriceModel()
    model.preprocess_data(df)

    # Update the GradientBoostingRegressor parameters
    model.model = GradientBoostingRegressor(
        n_estimators=500,      # More trees
        learning_rate=0.05,    # Slower learning rate for better generalization
        max_depth=6,          # Slightly deeper trees
        min_samples_split=5,   # Minimum samples required to split
        min_samples_leaf=4,    # Minimum samples required at leaf node
        subsample=0.8,        # Use 80% of samples for each tree
        random_state=42
    )

    model.train(df)
    
    # Save model
    model.save_model()
    
    # Example prediction
    sample_input = pd.DataFrame({
        'type': ['Car'],
        'brand': ['Toyota'],
        'model': ['Corolla'],
        'year_of_manufacture': [2020],
        'mileage_km': [50000],
        'transmission_type': ['Automatic'],
        'fuel_type': ['Petrol'],
        'accidents_history': ['None'],
        'owner': [1],
        'seats': [5],
        'horse_power': [132],
        'city': ['Casablanca']
    })
    
    prediction = model.predict(sample_input)
    print("\nSample Prediction:")
    print(f"Estimated Price: {prediction['prediction']:,} MAD")
    print(f"Confidence Interval: [{prediction['confidence_interval'][0]:,} - {prediction['confidence_interval'][1]:,}] MAD") 