# Vehicle Price Estimation Model

This project implements a machine learning model to estimate vehicle prices in Morocco (MAD currency).

## Available Brands and Models

### Cars

1. **Mercedes** (Premium Factor: 1.8)
   - Classic Series: 190, 200, 220, 250, 300
   - C-Class: C180, C200, C220, C250, C300
   - E-Class: E200, E220, E250, E300
   - SUV/Crossover: CLA, GLA, GLK

2. **Volkswagen** (Premium Factor: 1.3)
   - Hatchbacks: Golf, Polo, Golf Plus, Scirocco
   - Sedans: Passat, Jetta, Bora
   - MPV/SUV: Tiguan, Caddy, Touran

3. **Renault** (Premium Factor: 1.0)
   - Compact: Clio, Symbol
   - Mid-size: Megane, Fluence
   - MPV: Scenic, Kangoo
   - SUV: Captur, Kadjar
   - Other: Logan, Express, Laguna, Talisman

4. **Peugeot** (Premium Factor: 1.1)
   - Classic Series: 106, 205, 206, 207, 208
   - Mid-size: 301, 306, 307, 308, 309
   - Large: 404, 405, 406, 407, 408, 508
   - SUV: 2008, 3008, 5008
   - Commercial: Partner

5. **Dacia** (Premium Factor: 0.9)
   - Core Models: Logan, Sandero
   - SUV: Duster
   - MPV: Lodgy, Dokker
   - Electric: Spring
   - Special: Stepway

6. **Fiat** (Premium Factor: 1.0)
   - Classic: Uno, Palio, Siena
   - Modern: Punto, Tipo, Grande Punto
   - Special: 500, Panda, Linea, Bravo
   - Commercial: Doblo

7. **Citroen** (Premium Factor: 1.1)
   - C-Series: C2, C3, C4, C5
   - Premium: DS3, DS4
   - Special: C-Elysee, Saxo, ZX
   - Commercial: Berlingo
   - MPV: Xsara, Picasso

8. **Ford** (Premium Factor: 1.1)
   - Compact: Fiesta, Ka
   - Mid-size: Focus, Escort
   - Large: Fusion, Mondeo, Sierra
   - Commercial: Transit, Courier, Connect
   - SUV: EcoSport

9. **Hyundai** (Premium Factor: 1.1)
   - Compact: i10, i20, i30, Accent, Atos
   - Mid-size: Elantra, Getz
   - SUV: Tucson, Santa Fe
   - MPV: Matrix, H1
   - Large: Sonata

10. **Toyota** (Premium Factor: 1.2)
    - Compact: Yaris, Auris
    - Mid-size: Corolla
    - Large: Avensis, Camry
    - SUV: RAV4, Land Cruiser
    - MPV: Verso
    - Commercial: Hilux, Hiace
    - Hybrid: Prius

11. **Kia** (Premium Factor: 1.0)
    - Compact: Picanto, Rio
    - Mid-size: Cerato, Ceed
    - SUV: Sportage, Sorento
    - MPV: Carens
    - Large: Optima, Soul
    - Commercial: K2700

12. **Nissan** (Premium Factor: 1.2)
    - Compact: Micra, Note
    - Mid-size: Sunny, Almera, Tiida, Primera
    - SUV: Qashqai, X-Trail, Juke
    - Large: Patrol
    - Commercial: Navara

13. **Opel** (Premium Factor: 1.1)
    - Compact: Corsa, Kadett
    - Mid-size: Astra
    - Large: Vectra, Insignia, Omega
    - MPV: Meriva, Zafira
    - Commercial: Combo

14. **Seat** (Premium Factor: 1.1)
    - Compact: Ibiza, Cordoba
    - Mid-size: Leon, Toledo
    - MPV: Altea
    - SUV: Arona, Ateca
    - Commercial: Inca

15. **Skoda** (Premium Factor: 1.1)
    - Compact: Fabia, Felicia
    - Mid-size: Rapid
    - Large: Octavia, Superb
    - MPV: Roomster
    - SUV: Yeti, Karoq

16. **Chevrolet** (Premium Factor: 1.0)
    - Compact: Spark, Matiz
    - Mid-size: Aveo, Lacetti
    - Large: Cruze, Epica
    - MPV: Orlando
    - SUV: Captiva

17. **Suzuki** (Premium Factor: 1.0)
    - Compact: Alto, Swift, Baleno, Celerio, Ignis
    - SUV: Vitara, Jimny
    - Crossover: SX4

18. **Mitsubishi** (Premium Factor: 1.2)
    - Compact: Colt, Space Star
    - Mid-size: Lancer
    - SUV: Pajero, ASX, Outlander
    - Commercial: L200

### Motorcycles

1. **Honda** (Premium Factor: 1.3)
   - Sport: CBR125R, CB125F
   - Standard: CG125, CBF125, Bros
   - Off-road: XR125L
   - Scooter: Wave125, PCX125, SH125, Lead, Vision, Forza

2. **Yamaha** (Premium Factor: 1.3)
   - Sport: YZF-R125, MT-125
   - Standard: YBR125, YS125
   - Off-road: DTR125, WR125
   - Scooter: NMAX125, Crypton, Jog, Majesty

3. **BMW Motorrad** (Premium Factor: 1.8)
   - Adventure: G310GS, F750GS, F850GS, R1250GS
   - Sport: G310R, S1000RR
   - Heritage: R nineT
   - Urban Mobility: C400X

4. **Kawasaki** (Premium Factor: 1.3)
   - Sport: Ninja 125, Z125
   - Adventure: Versys, KLR
   - Cruiser: Vulcan
   - Off-road: KLX125, D-Tracker
   - Classic: W175

5. **KTM** (Premium Factor: 1.5)
   - Naked: 125 Duke, 200 Duke, 390 Duke
   - Sport: RC125, RC200, RC390
   - Adventure: Adventure 390
   - Supermoto: 690 SMC

6. **Suzuki** (Premium Factor: 1.2)
   - Sport: GSX-R125, GSX-S125
   - Cruiser: Boulevard, Intruder
   - Scooter: Address, Burgman
   - Adventure: V-Strom
   - Standard: Bandit

7. **Zontes** (Premium Factor: 1.2)
   - Sport: ZT125, ZT310, ZT350
   - Urban: U125, G1
   - Touring: T310
   - Adventure: R350, GK350

8. **Royal Enfield** (Premium Factor: 1.3)
   - Classic: Classic 350, Bullet 350
   - Modern Classic: Continental GT, Interceptor
   - Adventure: Himalayan
   - Cruiser: Meteor 350
   - Urban: Hunter

9. **Bajaj** (Premium Factor: 0.9)
   - Sport: Pulsar, Dominar
   - Cruiser: Avenger
   - Commuter: Platina, Boxer, Discover
   - Classic: V15

10. **Sym** (Premium Factor: 1.0)
    - Scooter: Orbit, Symphony, Jet, Fiddle, Joy
    - Standard: Wolf, Crox
    - Special: X-Pro, Mask, Allo

11. **Vespa** (Premium Factor: 1.4)
    - Classic: Primavera, Sprint, PX
    - Modern: GTS, LX, S125
    - Special: ET4, GT, GTV
    - Electric: Elettrica

12. **Kymco** (Premium Factor: 1.1)
    - Scooter: Agility, People, Like, Super8
    - Maxi-scooter: X-Town, Downtown
    - Special: Dink, Vitality, Grand Dink

13. **Benelli** (Premium Factor: 1.2)
    - Sport: TNT125
    - Standard: BN125, 125i, 150i, 165s
    - Scooter: Caffenero, Tornado, Velvet, Zafferano

14. **Piaggio** (Premium Factor: 1.3)
    - Scooter: Liberty, Medley, Beverly, Zip
    - Special: MP3, NRG, Typhoon, Fly

15. **Keeway** (Premium Factor: 0.9)
    - Sport: RKS, RKV
    - Cruiser: K-Light
    - Standard: Superlight
    - Scooter: Cityblade, Target, TX

16. **Aprilia** (Premium Factor: 1.4)
    - Sport: SR, RS
    - Scooter: Scarabeo, Sportcity
    - Special: Leonardo, Atlantic

17. **MBK** (Premium Factor: 1.0)
    - Sport: Booster, Stunt
    - Scooter: Nitro, Ovetto
    - Special: Flame, Skyliner

18. **Peugeot Motorcycles** (Premium Factor: 1.1)
    - Scooter: Kisbee, Django, Tweet
    - Sport: Speedfight
    - Special: Vivacity, Ludix

19. **Lifan** (Premium Factor: 0.8)
    - Standard: KP, LF
    - Off-road: X-Ride
    - Special: V16, Mirage, Hunter

## Project Structure
```
├── data/
│   ├── all_type_data.csv        # Original data
│   └── generated_data.csv       # Generated data with prices
├── src/
│   ├── data_generator.py        # Script to generate training data
│   ├── model_trainer.py         # ML model training script
│   └── utils.py                 # Utility functions
├── web/
│   ├── static/
│   │   ├── css/
│   │   └── js/
│   ├── templates/
│   └── app.py                   # Flask web application
├── models/                      # Saved model files
├── requirements.txt             # Python dependencies
└── README.md                   # This file
```

## Setup
1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python web/app.py
```

## Features
- Price estimation for cars and motorcycles
- Advanced interface with dropdowns for brands/models/cities
- Confidence interval visualization
- K-fold cross-validation
- Train-test-validation split (70-15-15)
- Time-based validation for vehicle age impact

## Model Details
- Uses multiple validation strategies
- Includes confidence intervals
- Handles various vehicle types and conditions
- City-specific pricing factors 