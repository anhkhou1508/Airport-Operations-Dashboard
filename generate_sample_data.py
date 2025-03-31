import random
import datetime
from sqlalchemy.orm import Session
import models
from database import engine, get_db
from models import FlightStatus
import pandas as pd
import numpy as np

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

# Sample airlines
AIRLINES = ["American", "Delta", "United", "Southwest", "JetBlue", "Alaska", "Spirit", "Frontier"]

# Sample airports
AIRPORTS = {
    "ATL": "Atlanta",
    "LAX": "Los Angeles",
    "ORD": "Chicago",
    "DFW": "Dallas-Fort Worth",
    "DEN": "Denver", 
    "JFK": "New York",
    "SFO": "San Francisco",
    "SEA": "Seattle",
    "LAS": "Las Vegas",
    "MCO": "Orlando",
    "MIA": "Miami",
    "BOS": "Boston",
    "PHX": "Phoenix",
    "CLT": "Charlotte",
    "IAH": "Houston"
}

def random_date(start_date, end_date):
    """Generate a random date between start_date and end_date."""
    time_delta = end_date - start_date
    random_days = random.randrange(time_delta.days)
    return start_date + datetime.timedelta(days=random_days)

def generate_flight_data(num_flights=100):
    """Generate random flight data."""
    flights = []
    
    # Define date range for flights
    start_date = datetime.datetime.now() - datetime.timedelta(days=30)
    end_date = datetime.datetime.now() + datetime.timedelta(days=30)
    
    for i in range(num_flights):
        # Generate a random departure date
        departure_date = random_date(start_date, end_date)
        
        # Select random airports
        departure_airport, arrival_airport = random.sample(list(AIRPORTS.keys()), 2)
        
        # Generate a random flight time (1-6 hours)
        flight_time = datetime.timedelta(hours=random.randint(1, 6), 
                                         minutes=random.randint(0, 59))
        
        # Set scheduled arrival
        scheduled_arrival = departure_date + flight_time
        
        # Generate flight number
        airline = random.choice(AIRLINES)
        flight_number = f"{airline[:2].upper()}{random.randint(1000, 9999)}"
        
        # Determine flight status based on departure date
        now = datetime.datetime.now()
        
        if departure_date > now + datetime.timedelta(hours=1):
            # Flight in the future
            status = FlightStatus.ON_TIME.value
        elif departure_date <= now < scheduled_arrival:
            # Flight in progress
            statuses = [FlightStatus.ON_TIME.value, FlightStatus.DELAYED.value, FlightStatus.BOARDING.value]
            status = random.choices(statuses, weights=[0.7, 0.2, 0.1])[0]
        else:
            # Flight in the past
            statuses = [FlightStatus.ARRIVED.value, FlightStatus.DELAYED.value, FlightStatus.CANCELLED.value]
            status = random.choices(statuses, weights=[0.75, 0.15, 0.1])[0]
            
            # If the flight was delayed, adjust the arrival time
            if status == FlightStatus.DELAYED.value:
                # Add a delay between 15 minutes and 3 hours
                delay_minutes = random.randint(15, 180)
                scheduled_arrival += datetime.timedelta(minutes=delay_minutes)
        
        # Generate a gate
        gate = f"{random.choice('ABCDEFG')}{random.randint(1, 30)}"
        
        flights.append({
            "flight_number": flight_number,
            "airline": airline,
            "departure_airport": departure_airport,
            "arrival_airport": arrival_airport,
            "scheduled_departure": departure_date,
            "scheduled_arrival": scheduled_arrival,
            "status": status,
            "gate": gate
        })
    
    return flights

def add_flights_to_db(flights, db: Session):
    """Add flights to the database."""
    for flight_data in flights:
        # Check if the flight already exists
        existing = db.query(models.Flight).filter(
            models.Flight.flight_number == flight_data["flight_number"]
        ).first()
        
        if not existing:
            # Create a new flight
            flight = models.Flight(**flight_data)
            db.add(flight)
    
    # Commit all changes
    db.commit()
    
    return len(flights)

def main():
    """Generate sample flight data and add it to the database."""
    # Generate random flight data
    flights = generate_flight_data(num_flights=200)
    
    # Get a database session
    db = next(get_db())
    
    try:
        # Add flights to the database
        num_added = add_flights_to_db(flights, db)
        print(f"Added {num_added} flights to the database.")
    finally:
        db.close()

if __name__ == "__main__":
    main() 