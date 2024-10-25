import pandas as pd
import itertools
import random

# Step 1: Define unique values
companies = ['Hyundai', 'Ford', 'Honda', 'KIA']
models = ['All New Santro', 'Creta', 'Grand i10', 'i20', 'Ecosport', 'Figo', 'Amaze', 'City (2014)', 'WR-V', 'Carnival']
fuels = ['Petrol', '1.1 Petrol', '1.2L Petrol', '1.5L Petrol', 'Diesel', '1.4L Diesel', '1.5L Diesel', '2.2L Diesel']
cities = ['Mumbai', 'Delhi', 'Srinagar', 'Shimla', 'Vishakhapattnam']
ages = range(0, 121)  # Vehicle age from 0 to 120 years

# Step 2: Generate all combinations
data = []
for company, model, fuel, city in itertools.product(companies, models, fuels, cities):
    for age in ages:
        # Step 3: Generate random values for remaining attributes
        mileage = random.randint(1000, 200000)  # Random mileage
        mileage_km_per_l = random.choice([25, 20, 15, 12, 10])  # Random km/l value
        # Simulate other components as random values or fixed values
        air_cleaner_filter = random.randint(0, 1)
        engine_oil = random.randint(800, 1200)
        engine_oil_filter = random.randint(50, 100)
        sump_plug_gasket = random.randint(0, 1)
        ac_dust_filter = random.randint(0, 1)
        climate_control_air_filter = random.randint(0, 1)
        fuel_filter = random.randint(0, 1)
        engine_coolant = random.randint(0, 1)
        spark_plug = random.randint(0, 1)
        brake_oil = random.randint(0, 1)
        brake_pad = random.randint(0, 1)
        windshield_wiper = random.randint(0, 1)
        clutch = random.randint(0, 1)
        battery = random.randint(0, 1)
        pollen_filter = random.randint(0, 1)
        transmission_fluid = random.randint(0, 1)
        drain_washer = random.randint(0, 1)
        total_labour_charges = random.randint(500, 5000)
        total_parts_charges = random.randint(1000, 10000)
        total_cost = total_labour_charges + total_parts_charges
        
        # Step 4: Append the generated row
        data.append([
            company, model, fuel, city, age, mileage, mileage_km_per_l,
            air_cleaner_filter, engine_oil, engine_oil_filter,
            sump_plug_gasket, ac_dust_filter, climate_control_air_filter,
            fuel_filter, engine_coolant, spark_plug, brake_oil,
            brake_pad, windshield_wiper, clutch, battery, pollen_filter,
            transmission_fluid, drain_washer, total_labour_charges,
            total_parts_charges, total_cost
        ])

# Create a DataFrame and save to CSV
columns = [
    'Company', 'Model', 'Fuel', 'City', 'Age of Vehicle', 'Mileage', 
    'Mileage (km/l)', 'Air cleaner filter', 'Engine oil', 
    'Engine oil filter', 'Sump Plug Gasket', 'AC Dust Filter', 
    'Climate control air filter', 'Fuel filter', 'Engine coolant', 
    'Spark plug', 'Brake oil', 'Brake pad', 'Windshield wiper', 
    'Clutch', 'Battery', 'Pollen Filter', 'Transmission fluid', 
    'Drain Washer', 'Total Labour charges', 'Total parts charges', 
    'Total cost'
]

# Create a DataFrame from the data
df = pd.DataFrame(data, columns=columns)

# Save to a CSV file
df.to_csv('generated_vehicle_data.csv', index=False)

print("Dataset created successfully and saved to 'generated_vehicle_data.csv'.")
