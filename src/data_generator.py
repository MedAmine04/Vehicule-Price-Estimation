import pandas as pd
import numpy as np
from datetime import datetime
import os

# Get the absolute path to the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Constants
CITIES = [
    'Casablanca', 'Rabat', 'Marrakech', 'Tangier', 'Fes', 'Agadir',
    'Meknes', 'Oujda', 'Kenitra', 'Tetouan', 'El Jadida', 'Safi',
    'Mohammedia', 'Khouribga', 'Beni Mellal', 'Nador', 'Taza'
]

CURRENT_YEAR = datetime.now().year

# Car brands and their models with premium factors (1.0 = standard, >1.0 = premium)
CAR_BRANDS_MODELS = {
    'Mercedes': {
        'premium_factor': 1.8,
        'models': ['190', '200', '220', '250', '300', 'C180', 'C200', 'C220', 'C250', 'C300', 'E200', 'E220', 'E250', 'E300', 'CLA', 'GLA', 'GLK', 'GLC', 'GLE', 'A180', 'A200']
    },
    'Volkswagen': {
        'premium_factor': 1.3,
        'models': ['Golf', 'Polo', 'Passat', 'Tiguan', 'Caddy', 'Touran', 'Jetta', 'Bora', 'Golf Plus', 'Scirocco', 'T-Roc', 'T-Cross', 'ID.4', 'ID.3']
    },
    'Renault': {
        'premium_factor': 1.0,
        'models': ['Clio', 'Megane', 'Symbol', 'Logan', 'Kangoo', 'Express', 'Fluence', 'Scenic', 'Captur', 'Laguna', 'Talisman', 'Kadjar', 'Arkana', 'Austral']
    },
    'Peugeot': {
        'premium_factor': 1.1,
        'models': ['106', '205', '206', '207', '208', '301', '306', '307', '308', '309', '404', '405', '406', '407', '408', '508', '2008', '3008', '5008', 'Partner', 'Rifter']
    },
    'Dacia': {
        'premium_factor': 0.9,
        'models': ['Logan', 'Sandero', 'Duster', 'Lodgy', 'Dokker', 'Spring', 'Stepway', 'Jogger']
    },
    'Fiat': {
        'premium_factor': 1.0,
        'models': ['Uno', 'Palio', 'Siena', 'Punto', 'Tipo', 'Doblo', '500', 'Panda', 'Linea', 'Grande Punto', 'Bravo', 'Ducato', '500X', '500L']
    },
    'Citroen': {
        'premium_factor': 1.1,
        'models': ['C3', 'C4', 'C5', 'Berlingo', 'Xsara', 'Picasso', 'DS3', 'DS4', 'C-Elysee', 'Saxo', 'ZX', 'C2', 'C3 Aircross', 'C5 Aircross']
    },
    'Ford': {
        'premium_factor': 1.1,
        'models': ['Fiesta', 'Focus', 'Escort', 'Fusion', 'Mondeo', 'Transit', 'Courier', 'Ka', 'Sierra', 'Connect', 'EcoSport', 'Puma', 'Kuga']
    },
    'Hyundai': {
        'premium_factor': 1.1,
        'models': ['Accent', 'i10', 'i20', 'i30', 'Elantra', 'Tucson', 'Santa Fe', 'H1', 'Atos', 'Getz', 'Matrix', 'Sonata', 'Kona', 'Bayon']
    },
    'Toyota': {
        'premium_factor': 1.2,
        'models': ['Corolla', 'Yaris', 'Avensis', 'Camry', 'RAV4', 'Hilux', 'Land Cruiser', 'Hiace', 'Auris', 'Verso', 'Prius', 'C-HR', 'Yaris Cross', 'bZ4X']
    },
    'Kia': {
        'premium_factor': 1.0,
        'models': ['Picanto', 'Rio', 'Cerato', 'Sportage', 'Sorento', 'K2700', 'Carens', 'Optima', 'Soul', 'Ceed', 'XCeed', 'Niro']
    },
    'Nissan': {
        'premium_factor': 1.2,
        'models': ['Micra', 'Sunny', 'Almera', 'Tiida', 'Qashqai', 'X-Trail', 'Patrol', 'Navara', 'Juke', 'Note', 'Primera', 'Ariya', 'Leaf']
    },
    'Opel': {
        'premium_factor': 1.1,
        'models': ['Corsa', 'Astra', 'Vectra', 'Insignia', 'Meriva', 'Zafira', 'Combo', 'Kadett', 'Omega', 'Mokka', 'Crossland', 'Grandland']
    },
    'Seat': {
        'premium_factor': 1.1,
        'models': ['Ibiza', 'Leon', 'Toledo', 'Altea', 'Cordoba', 'Arona', 'Ateca', 'Inca', 'Tarraco', 'Born']
    },
    'Skoda': {
        'premium_factor': 1.1,
        'models': ['Fabia', 'Octavia', 'Superb', 'Rapid', 'Roomster', 'Yeti', 'Karoq', 'Felicia', 'Scala', 'Kamiq', 'Enyaq']
    },
    'Chevrolet': {
        'premium_factor': 1.0,
        'models': ['Spark', 'Aveo', 'Cruze', 'Captiva', 'Orlando', 'Epica', 'Lacetti', 'Matiz', 'Trax', 'Malibu']
    },
    'Suzuki': {
        'premium_factor': 1.0,
        'models': ['Alto', 'Swift', 'Vitara', 'Jimny', 'SX4', 'Baleno', 'Celerio', 'Ignis', 'S-Cross', 'Across']
    },
    'Mitsubishi': {
        'premium_factor': 1.2,
        'models': ['Lancer', 'Pajero', 'L200', 'ASX', 'Outlander', 'Space Star', 'Colt', 'Eclipse Cross']
    },
    'BYD': {
        'premium_factor': 1.4,
        'models': ['Han', 'Tang', 'Song', 'Yuan', 'Dolphin', 'Seal', 'Atto 3', 'Seal U', 'Seal U PHEV']
    },
    'Tesla': {
        'premium_factor': 2.0,
        'models': ['Model 3', 'Model Y', 'Model S', 'Model X']
    },
    'Audi': {
        'premium_factor': 1.7,
        'models': ['A1', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'Q2', 'Q3', 'Q5', 'Q7', 'Q8', 'e-tron', 'TT']
    },
    'BMW': {
        'premium_factor': 1.7,
        'models': ['1 Series', '2 Series', '3 Series', '4 Series', '5 Series', '7 Series', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'i3', 'i4', 'iX']
    }
}

# Motorcycle brands and their models
MOTORCYCLE_BRANDS_MODELS = {
    'Honda': {
        'premium_factor': 1.3,
        'models': ['CG125', 'CBF125', 'CB125F', 'Wave125', 'Bros', 'XR125L', 'CBR125R', 'PCX125', 'SH125', 'Lead', 'Vision', 'Forza']
    },
    'Yamaha': {
        'premium_factor': 1.3,
        'models': ['YBR125', 'YS125', 'MT-125', 'NMAX125', 'DTR125', 'WR125', 'YZF-R125', 'Crypton', 'Jog', 'Majesty']
    },
    'Sym': {
        'premium_factor': 1.0,
        'models': ['Orbit', 'Symphony', 'Jet', 'Crox', 'Wolf', 'Fiddle', 'Joy', 'X-Pro', 'Mask', 'Allo']
    },
    'Vespa': {
        'premium_factor': 1.4,
        'models': ['Primavera', 'Sprint', 'GTS', 'LX', 'S125', 'PX', 'ET4', 'GT', 'GTV', 'Elettrica']
    },
    'Kymco': {
        'premium_factor': 1.1,
        'models': ['Agility', 'People', 'Like', 'Super8', 'X-Town', 'Downtown', 'Dink', 'Vitality', 'Grand Dink']
    },
    'Benelli': {
        'premium_factor': 1.2,
        'models': ['TNT125', 'BN125', '125i', '150i', '165s', 'Caffenero', 'Tornado', 'Velvet', 'Zafferano']
    },
    'Piaggio': {
        'premium_factor': 1.3,
        'models': ['Liberty', 'Medley', 'Beverly', 'MP3', 'Zip', 'NRG', 'Typhoon', 'Fly']
    },
    'Keeway': {
        'premium_factor': 0.9,
        'models': ['RKS', 'RKV', 'Superlight', 'K-Light', 'Cityblade', 'Target', 'TX']
    },
    'Aprilia': {
        'premium_factor': 1.4,
        'models': ['SR', 'RS', 'Scarabeo', 'Sportcity', 'Leonardo', 'Atlantic']
    },
    'MBK': {
        'premium_factor': 1.0,
        'models': ['Booster', 'Nitro', 'Ovetto', 'Stunt', 'Flame', 'Skyliner']
    },
    'Peugeot': {
        'premium_factor': 1.1,
        'models': ['Kisbee', 'Speedfight', 'Django', 'Tweet', 'Vivacity', 'Ludix']
    },
    'Lifan': {
        'premium_factor': 0.8,
        'models': ['KP', 'LF', 'X-Ride', 'V16', 'Mirage', 'Hunter']
    },
    'Zontes': {
        'premium_factor': 1.2,
        'models': ['ZT125', 'ZT310', 'ZT350', 'U125', 'G1', 'T310', 'R350', 'GK350']
    },
    'Suzuki': {
        'premium_factor': 1.2,
        'models': ['GSX-R125', 'GSX-S125', 'Address', 'Burgman', 'V-Strom', 'Boulevard', 'Intruder', 'Bandit']
    },
    'Kawasaki': {
        'premium_factor': 1.3,
        'models': ['Ninja 125', 'Z125', 'Versys', 'Vulcan', 'KLX125', 'D-Tracker', 'W175', 'KLR']
    },
    'Royal Enfield': {
        'premium_factor': 1.3,
        'models': ['Classic 350', 'Meteor 350', 'Bullet 350', 'Himalayan', 'Continental GT', 'Interceptor', 'Hunter']
    },
    'Bajaj': {
        'premium_factor': 0.9,
        'models': ['Pulsar', 'Dominar', 'Avenger', 'Platina', 'Boxer', 'Discover', 'V15']
    }
}

def generate_base_price(vehicle_type, brand_factor, year, mileage, accidents, owners, hp):
    """Generate base price in MAD based on vehicle characteristics"""
    # Base price ranges
    if vehicle_type == 'Car':
        base = np.random.uniform(50000, 200000)  # Base for economy car
        if brand_factor > 1.3:
            base = np.random.uniform(200000, 500000)  # Mid-range
        if brand_factor > 1.6:
            base = np.random.uniform(500000, 2000000)  # Luxury
    else:  # Motorcycle
        base = np.random.uniform(8000, 15000)  # Standard motorcycle
        if brand_factor > 1.2:
            base = np.random.uniform(15000, 30000)  # Premium motorcycle

    # Apply factors
    age_factor = 0.93 ** (CURRENT_YEAR - year)  # 7% depreciation per year
    mileage_factor = 0.9 ** (mileage / 50000)  # Depreciation per 50,000 km
    accident_factor = 0.7 if accidents == 'Major' else 0.85 if accidents == 'Minor' else 1.0
    owner_factor = 0.95 ** (owners - 1)  # 5% reduction per previous owner
    hp_factor = 1 + (hp / 1000)  # Small increase for more powerful vehicles

    price = base * brand_factor * age_factor * mileage_factor * accident_factor * owner_factor * hp_factor
    
    # Add some random variation (±10%)
    price *= np.random.uniform(0.9, 1.1)
    
    return round(price, -3)  # Round to nearest 1000

def generate_dataset(n_samples=2000):
    """Generate a dataset with n_samples records"""
    data = []
    
    for _ in range(n_samples):
        # Determine vehicle type
        vehicle_type = np.random.choice(['Car', 'Motorcycle'], p=[0.7, 0.3])
        
        # Select brand and model
        if vehicle_type == 'Car':
            brand = np.random.choice(list(CAR_BRANDS_MODELS.keys()))
            brand_data = CAR_BRANDS_MODELS[brand]
            model = np.random.choice(brand_data['models'])
            brand_factor = brand_data['premium_factor']
            seats = np.random.randint(2, 8) if vehicle_type == 'Car' else 2
            fuel = np.random.choice(['Petrol', 'Diesel', 'Electric', 'Hybrid'])
            hp = np.random.randint(70, 300)  # Car horsepower range
        else:
            brand = np.random.choice(list(MOTORCYCLE_BRANDS_MODELS.keys()))
            brand_data = MOTORCYCLE_BRANDS_MODELS[brand]
            model = np.random.choice(brand_data['models'])
            brand_factor = brand_data['premium_factor']
            seats = 2
            fuel = np.random.choice(['Petrol', 'Electric'])
            hp = np.random.randint(8, 25)  # Motorcycle horsepower range
        
        # Generate other features
        year = np.random.randint(2000, CURRENT_YEAR + 1)
        mileage = np.random.randint(0, 300000)
        transmission = np.random.choice(['Manual', 'Automatic'])
        accidents = np.random.choice(['None', 'Minor', 'Major'], p=[0.7, 0.2, 0.1])
        owners = np.random.randint(1, 6)
        city = np.random.choice(CITIES)
        
        # Generate price
        price = generate_base_price(vehicle_type, brand_factor, year, mileage, accidents, owners, hp)
        
        # Create record
        record = {
            'type': vehicle_type,
            'brand': brand,
            'model': model,
            'year_of_manufacture': year,
            'mileage_km': mileage,
            'transmission_type': transmission,
            'fuel_type': fuel,
            'accidents_history': accidents,
            'owner': owners,
            'seats': seats,
            'horse_power': hp,
            'city': city,
            'price_mad': price
        }
        data.append(record)
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    # Generate dataset
    df = generate_dataset(2000)
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save to CSV
    output_file = os.path.join(data_dir, 'generated_data.csv')
    df.to_csv(output_file, index=False)
    print(f"Generated {len(df)} records with prices in MAD")
    print("\nSample of the generated data:")
    print(df.head())
    print("\nSummary statistics:")
    print(df.describe()) 
    print(df.describe()) 