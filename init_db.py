import logging
from sqlalchemy.orm import Session
from database import engine, SessionLocal, Base
import models
from datetime import datetime, timedelta
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_db():
    # Create tables
    logger.info("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    
    db = SessionLocal()
    try:
        # Check if we already have flights
        flight_count = db.query(models.Flight).count()
        if flight_count > 0:
            logger.info(f"Database already contains {flight_count} flights. Skipping seed data.")
            return
            
        # Generate seed data
        logger.info("Seeding database with sample flights...")
        airlines = ["Delta", "United", "American", "Southwest", "British Airways", "Lufthansa", "Emirates"]
        statuses = [status.value for status in models.FlightStatus]
        airports = ["JFK", "LAX", "ORD", "ATL", "LHR", "CDG", "DXB", "HND", "SFO"]
        aircraft_types = ["Boeing 737-800", "Airbus A320", "Boeing 787-9", "Airbus A321", "Boeing 777-300", "Airbus A350"]
        
        now = datetime.now()
        
        # Create 20 sample flights
        created_flights = []
        for i in range(20):
            # Generate random departure time within the next 24 hours
            departure_time = now + timedelta(hours=random.randint(0, 24), minutes=random.randint(0, 59))
            # Generate random arrival time 1-8 hours after departure
            arrival_time = departure_time + timedelta(hours=random.randint(1, 8), minutes=random.randint(0, 59))
            
            # Select departure and arrival airports
            departure_airport = random.choice(airports)
            # Make sure arrival is different from departure
            arrival_airport = random.choice([a for a in airports if a != departure_airport])
            
            flight = models.Flight(
                flight_number=f"{random.choice(['A', 'B', 'D', 'U', 'E'])}{random.randint(100, 999)}",
                airline=random.choice(airlines),
                departure_airport=departure_airport,
                arrival_airport=arrival_airport,
                scheduled_departure=departure_time,
                scheduled_arrival=arrival_time,
                status=random.choice(statuses),
                gate=f"{random.choice(['A', 'B', 'C', 'D', 'E'])}{random.randint(1, 30)}",
                # Add new fields
                aircraft_type=random.choice(aircraft_types),
                aircraft_age=random.uniform(0.5, 15.0),  # Aircraft age in years
            )
            db.add(flight)
            created_flights.append(flight)
        
        # Commit the flights first to get IDs
        db.commit()
        
        # Now add some previous flight relationships (for previous leg delays)
        for i in range(5):  # Link a few flights as previous legs
            if len(created_flights) >= 2:
                flight1, flight2 = random.sample(created_flights, 2)
                # Ensure flight1 is before flight2
                if flight1.scheduled_arrival < flight2.scheduled_departure:
                    flight2.previous_flight_id = flight1.id
                    # If flight1 is delayed, add delay value
                    if "Delayed" in flight1.status:
                        flight2.previous_flight_delay = random.randint(15, 120)  # Delay in minutes
                        
        # Add some weather data
        logger.info("Adding sample weather data...")
        for flight in created_flights:
            # Departure weather
            db.add(models.WeatherCondition(
                flight_id=flight.id,
                airport_code=flight.departure_airport,
                timestamp=flight.scheduled_departure,
                temperature=random.uniform(0, 35),  # Celsius
                wind_speed=random.uniform(0, 30),   # km/h
                precipitation=random.uniform(0, 25),  # mm
                visibility=random.uniform(0.5, 15),  # km
                storm_indicator=random.random() < 0.1,  # 10% chance of storm
                is_departure=True
            ))
            
            # Arrival weather
            db.add(models.WeatherCondition(
                flight_id=flight.id,
                airport_code=flight.arrival_airport,
                timestamp=flight.scheduled_arrival,
                temperature=random.uniform(0, 35),
                wind_speed=random.uniform(0, 30),
                precipitation=random.uniform(0, 25),
                visibility=random.uniform(0.5, 15),
                storm_indicator=random.random() < 0.1,
                is_departure=False
            ))
        
        # Add airport congestion data
        logger.info("Adding sample airport congestion data...")
        for airport in airports:
            # Add congestion data for different times
            for hour in range(0, 24, 3):  # Every 3 hours
                time_point = now.replace(hour=hour, minute=0, second=0) + timedelta(days=random.randint(0, 2))
                
                # Peak hours have higher congestion
                if 6 <= hour <= 9 or 16 <= hour <= 19:
                    congestion_level = random.uniform(0.7, 0.95)
                    departures = random.randint(30, 60)
                    arrivals = random.randint(30, 60)
                else:
                    congestion_level = random.uniform(0.2, 0.6)
                    departures = random.randint(5, 25)
                    arrivals = random.randint(5, 25)
                
                db.add(models.AirportCongestion(
                    airport_code=airport,
                    timestamp=time_point,
                    departures_count=departures,
                    arrivals_count=arrivals,
                    congestion_level=congestion_level
                ))
        
        # Add some FAA restrictions
        logger.info("Adding sample FAA restrictions...")
        for airport in random.sample(airports, 3):  # Random 3 airports
            start_time = now + timedelta(hours=random.randint(0, 48))
            end_time = start_time + timedelta(hours=random.randint(1, 4))
            
            db.add(models.AirTrafficRestriction(
                airport_code=airport,
                start_time=start_time,
                end_time=end_time,
                restriction_type=random.choice(["Ground Stop", "Ground Delay", "Airspace Flow Program"]),
                reason=random.choice(["Weather", "Volume", "Staffing", "Runway Construction"])
            ))
        
        # Add some holidays
        logger.info("Adding sample holidays...")
        holidays = [
            ("New Year's Day", datetime(now.year, 1, 1), "US"),
            ("Independence Day", datetime(now.year, 7, 4), "US"),
            ("Thanksgiving", datetime(now.year, 11, 25), "US"),
            ("Christmas", datetime(now.year, 12, 25), "US"),
        ]
        
        for name, date, country in holidays:
            db.add(models.HolidayEvent(
                date=date,
                name=name,
                is_holiday=True,
                country=country
            ))
        
        # Add historical performance data
        logger.info("Adding sample historical performance data...")
        for airline in airlines:
            db.add(models.HistoricalPerformance(
                airline=airline,
                avg_delay=random.uniform(5, 30),
                delay_frequency=random.uniform(0.05, 0.3),
                sample_size=random.randint(1000, 10000)
            ))
        
        for airport in airports:
            db.add(models.HistoricalPerformance(
                departure_airport=airport,
                avg_delay=random.uniform(5, 40),
                delay_frequency=random.uniform(0.05, 0.4),
                sample_size=random.randint(1000, 15000)
            ))
        
            db.add(models.HistoricalPerformance(
                arrival_airport=airport,
                avg_delay=random.uniform(5, 30),
                delay_frequency=random.uniform(0.05, 0.3),
                sample_size=random.randint(1000, 15000)
            ))
        
        # Add some route-specific data
        for _ in range(10):
            airline = random.choice(airlines)
            dep = random.choice(airports)
            arr = random.choice([a for a in airports if a != dep])
            
            db.add(models.HistoricalPerformance(
                airline=airline,
                departure_airport=dep,
                arrival_airport=arr,
                avg_delay=random.uniform(5, 30),
                delay_frequency=random.uniform(0.05, 0.3),
                sample_size=random.randint(500, 5000)
            ))
            
        db.commit()
        logger.info("Database initialization complete.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        db.rollback()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    init_db() 