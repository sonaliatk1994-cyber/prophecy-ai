import json
import time
import random
import os
from datetime import datetime, timedelta
from kafka import KafkaProducer
import pandas as pd

# config
BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
INTERVAL = int(os.getenv("GENERATE_INTERVAL", "5"))
TOPIC = "raw-properties"

# dubai areas with realistic price ranges
AREAS = {
    "Dubai Marina": {"rent_sqft": (65, 95), "sale_sqft": (1100, 1600), "demand": 0.85},
    "Downtown Dubai": {"rent_sqft": (80, 120), "sale_sqft": (1400, 2000), "demand": 0.92},
    "Palm Jumeirah": {"rent_sqft": (90, 150), "sale_sqft": (1800, 2500), "demand": 0.78},
    "JLT": {"rent_sqft": (50, 75), "sale_sqft": (800, 1100), "demand": 0.71},
    "Arabian Ranches": {"rent_sqft": (40, 60), "sale_sqft": (900, 1200), "demand": 0.65},
    "Bluewaters": {"rent_sqft": (85, 130), "sale_sqft": (1600, 2200), "demand": 0.80},
}

PROP_TYPES = ["Apartment", "Villa", "Townhouse", "Penthouse"]
AMENITIES_POOL = ["pool", "gym", "parking", "security", "balcony", "sea_view", "metro", "mall"]

def generate_property():
    area = random.choice(list(AREAS.keys()))
    area_data = AREAS[area]
    prop_type = random.choice(PROP_TYPES)

    # bedrooms based on type
    if prop_type == "Studio":
        beds = 0
    elif prop_type == "Penthouse":
        beds = random.choice([3,4,5])
    elif prop_type == "Villa":
        beds = random.choice([3,4,5,6])
    else:
        beds = random.choice([1,2,3])

    baths = max(1, beds + random.randint(-1, 2))
    sqft = beds * random.randint(350, 550) + random.randint(100, 400)
    if sqft < 400: sqft = 400

    floor = random.randint(1, 45) if prop_type in ["Apartment", "Penthouse"] else 1

    # pricing with some noise
    rent_price = int(sqft * random.uniform(*area_data["rent_sqft"]) * 12)  # annual
    sale_price = int(sqft * random.uniform(*area_data["sale_sqft"]))

    # days on market - higher demand = less days
    base_days = int((1 - area_data["demand"]) * 60) + random.randint(0, 30)

    # amenities
    num_amenities = random.randint(2, 6)
    amenities = random.sample(AMENITIES_POOL, num_amenities)

    # timestamp with seasonal factor
    now = datetime.now()
    # add slight seasonal bias (summer = lower demand in Dubai)
    month = now.month
    seasonal_factor = 1.0 if month not in [6,7,8] else 0.85

    property_obj = {
        "id": f"PROP-{random.randint(100000, 999999)}",
        "area": area,
        "property_type": prop_type,
        "bedrooms": beds,
        "bathrooms": baths,
        "sqft": sqft,
        "floor": floor,
        "rent_price_aed": rent_price,
        "sale_price_aed": sale_price,
        "days_on_market": base_days,
        "amenities": amenities,
        "listing_date": (now - timedelta(days=base_days)).isoformat(),
        "timestamp": now.isoformat(),
        "seasonal_factor": round(seasonal_factor, 2)
    }

    return property_obj

def main():
    print("[producer] starting up... waiting for kafka")
    time.sleep(15)  # wait for kafka to be ready

    producer = KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        key_serializer=lambda k: k.encode('utf-8') if k else None
    )

    print(f"[producer] connected to {BOOTSTRAP_SERVERS}, sending to topic '{TOPIC}'")

    count = 0
    try:
        while True:
            prop = generate_property()
            key = prop["area"].replace(" ", "_").lower()
            producer.send(TOPIC, key=key, value=prop)
            count += 1

            if count % 50 == 0:
                print(f"[producer] sent {count} messages so far...")

            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("\n[producer] shutting down")
    finally:
        producer.flush()
        producer.close()

if __name__ == "__main__":
    main()
