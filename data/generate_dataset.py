import pandas as pd
import numpy as np
import random
import string
import os
def generate_public_reference(size=8):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=size))

def generate_synthetic_data(num_rows=5000):
    colours = ['Red', 'Blue', 'Black', 'White', 'Silver', 'Grey', 'Green', 'Yellow', 'Brown', 'Orange']
    makes = ['Toyota', 'Ford', 'BMW', 'Audi', 'Honda', 'Chevrolet', 'Nissan', 'Kia', 'Hyundai', 'Volkswagen']
    models = {
        'Toyota': ['Corolla', 'Camry', 'RAV4'],
        'Ford': ['F-150', 'Focus', 'Escape'],
        'BMW': ['X5', '3 Series', '5 Series'],
        'Audi': ['A4', 'Q5', 'A6'],
        'Honda': ['Civic', 'Accord', 'CR-V'],
        'Chevrolet': ['Malibu', 'Silverado', 'Equinox'],
        'Nissan': ['Altima', 'Rogue', 'Sentra'],
        'Kia': ['Sportage', 'Optima', 'Sorento'],
        'Hyundai': ['Elantra', 'Tucson', 'Sonata'],
        'Volkswagen': ['Golf', 'Passat', 'Tiguan']
    }
    conditions = ['New', 'Used', 'Certified Used']
    body_types = ['Sedan', 'SUV', 'Hatchback', 'Convertible', 'Truck', 'Coupe']
    fuel_types = ['Petrol', 'Diesel', 'Electric', 'Hybrid']

    data = []
    for _ in range(num_rows):
        make = random.choice(makes)
        model = random.choice(models[make])
        mileage = random.randint(0, 250000)
        year = random.randint(2000, 2024)
        condition = random.choice(conditions)
        base_price = max(1000, (50000 - mileage * 0.1 + (year - 2000) * 500))
        if condition == 'Used':
            base_price *= 0.7
        elif condition == 'Certified Used':
            base_price *= 0.85
        base_price += random.uniform(-2000, 2000)

        price = max(500, int(base_price))

        entry = {
            'public_reference': generate_public_reference(),
            'mileage': mileage,
            'standard_colour': random.choice(colours),
            'standard_make': make,
            'standard_model': model,
            'vehicle_condition': condition,
            'year_of_registration': year,
            'price': price,
            'body_type': random.choice(body_types),
            'crossover_car_and_van': random.choice([True, False]),
            'fuel_type': random.choice(fuel_types)
        }
        data.append(entry)

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    df = generate_synthetic_data()
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, 'car_prices_synthetic.csv'), index=False)

    print("Synthetic dataset generated and saved to 'data/car_prices_synthetic.csv'.")
