

import requests
import pandas as pd
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dotenv import load_dotenv

load_dotenv()

# API Keys 
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
FLIGHTAWARE_API_KEY = os.getenv("FLIGHTAWARE_API_KEY", "")
AVIATION_STACK_API_KEY = os.getenv("AVIATION_STACK_API_KEY", "")

# Cache directory for storing API responses
CACHE_DIR = "data_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_weather_data(airport_code: str, timestamp: datetime) -> Dict[str, Any]:
    # Check cache first
    cache_file = os.path.join(CACHE_DIR, f"weather_{airport_code}_{timestamp.strftime('%Y%m%d')}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            pass  # If error reading cache, continue to API

    airport_coords = get_airport_coordinates(airport_code)
    if not airport_coords:
        return default_weather_data()
    
    if OPENWEATHER_API_KEY:
        lat, lon = airport_coords
        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                weather_data = {
                    "temperature": data["main"]["temp"],
                    "wind_speed": data["wind"]["speed"],
                    "precipitation": data.get("rain", {}).get("1h", 0),
                    "visibility": data.get("visibility", 10000) / 1000,  # Convert to km
                    "storm_indicator": "thunderstorm" in data.get("weather", [{}])[0].get("description", "").lower()
                }
                
                # Cache the result
                with open(cache_file, 'w') as f:
                    json.dump(weather_data, f)
                
                return weather_data
        except Exception as e:
            print(f"Error getting weather data: {str(e)}")
    
    return default_weather_data()

def get_airport_congestion(airport_code: str, timestamp: datetime) -> Dict[str, Union[int, float]]:
    # Check cache first
    cache_file = os.path.join(CACHE_DIR, f"congestion_{airport_code}_{timestamp.strftime('%Y%m%d%H')}.json")
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except:
            pass
    

    if FLIGHTAWARE_API_KEY:
        # Implementation would depend on the API
        pass
    
    # Return simulated congestion data based on time of day
    hour = timestamp.hour
    # Busier in morning and evening hours
    if 6 <= hour <= 9 or 16 <= hour <= 19:
        congestion_level = 0.8
        flights_count = 45
    # Medium in midday
    elif 10 <= hour <= 15:
        congestion_level = 0.6
        flights_count = 30
    # Low at night
    else:
        congestion_level = 0.3
        flights_count = 15
    
    # Add some randomness
    import random
    congestion_level = min(1.0, max(0.1, congestion_level + random.uniform(-0.1, 0.1)))
    flights_count = max(5, int(flights_count * random.uniform(0.8, 1.2)))
    
    congestion_data = {
        "congestion_level": congestion_level,
        "departures_count": flights_count // 2,
        "arrivals_count": flights_count // 2
    }
    
    # Cache the result
    with open(cache_file, 'w') as f:
        json.dump(congestion_data, f)
    
    return congestion_data

def get_aircraft_info(airline: str, flight_number: str) -> Dict[str, Any]:

    # Simulated aircraft data based on airline
    aircraft_data = {
        "Delta": {"type": "Boeing 737-800", "age": 7.5},
        "American": {"type": "Airbus A321", "age": 5.2},
        "United": {"type": "Boeing 787-9", "age": 4.8},
        "Southwest": {"type": "Boeing 737-700", "age": 12.3},
        "JetBlue": {"type": "Airbus A320", "age": 8.1},
        "Alaska": {"type": "Boeing 737-900", "age": 6.7},
        "Spirit": {"type": "Airbus A320neo", "age": 3.2},
        "Frontier": {"type": "Airbus A321neo", "age": 2.5},
    }
    
    return aircraft_data.get(airline, {"type": "Unknown", "age": 10.0})

def get_historical_performance(airline: str, departure: str, arrival: str, 
                               day_of_week: int, hour: int) -> Dict[str, float]:

    
    # Simulated data 
    weekend = day_of_week >= 5
    peak_hour = 7 <= hour <= 9 or 16 <= hour <= 19
    
    # Base delay values
    airline_delay = 10.0  # minutes
    departure_delay = 8.0
    arrival_delay = 7.0
    route_delay = 12.0
    
    # Adjust for weekend
    if weekend:
        airline_delay *= 1.2
        departure_delay *= 1.1
        arrival_delay *= 1.1
        route_delay *= 1.15
    
    # Adjust for peak hours
    if peak_hour:
        airline_delay *= 1.3
        departure_delay *= 1.25
        arrival_delay *= 1.2
        route_delay *= 1.35
    
    # Add some randomness
    import random
    airline_delay *= random.uniform(0.8, 1.2)
    departure_delay *= random.uniform(0.8, 1.2)
    arrival_delay *= random.uniform(0.8, 1.2)
    route_delay *= random.uniform(0.8, 1.2)
    
    return {
        "airline_delay": airline_delay,
        "departure_delay": departure_delay,
        "arrival_delay": arrival_delay,
        "route_delay": route_delay
    }

def get_faa_restrictions(airport_code: str, timestamp: datetime) -> List[Dict[str, Any]]:

    
    # Simulated data - 10% chance of a restriction
    import random
    if random.random() < 0.1:
        return [{
            "airport_code": airport_code,
            "start_time": (timestamp - timedelta(hours=1)).isoformat(),
            "end_time": (timestamp + timedelta(hours=2)).isoformat(),
            "restriction_type": random.choice(["Ground Stop", "Ground Delay", "Airspace Flow Program"]),
            "reason": random.choice(["Weather", "Volume", "Staffing", "Runway Construction"])
        }]
    
    return []

def is_holiday(date: datetime) -> bool:

    # Simple list of US holidays 
    year = date.year
    holidays = [
        f"{year}-01-01",  # New Year's Day
        f"{year}-07-04",  # Independence Day
        f"{year}-11-11",  # Veterans Day
        f"{year}-12-25",  # Christmas
        # Add more holidays
    ]
    
    return date.strftime("%Y-%m-%d") in holidays

def default_weather_data() -> Dict[str, Any]:
    """
    Return default weather data when API fails.
    """
    return {
        "temperature": 20.0,  # Celsius
        "wind_speed": 10.0,   # km/h
        "precipitation": 0.0,  # mm
        "visibility": 10.0,    # km
        "storm_indicator": False
    }

def get_airport_coordinates(airport_code: str) -> Optional[tuple]:

    airport_coords = {
        "ATL": (33.6407, -84.4277),   # Atlanta
        "LAX": (33.9416, -118.4085),  # Los Angeles
        "ORD": (41.9742, -87.9073),   # Chicago O'Hare
        "DFW": (32.8998, -97.0403),   # Dallas/Fort Worth
        "DEN": (39.8561, -104.6737),  # Denver
        "JFK": (40.6413, -73.7781),   # New York JFK
        "SFO": (37.7749, -122.4194),  # San Francisco
        "SEA": (47.4502, -122.3088),  # Seattle
        "LAS": (36.0840, -115.1537),  # Las Vegas
        "MCO": (28.4312, -81.3081),   # Orlando
    }
    
    return airport_coords.get(airport_code)

# Main function to gather all external data for a flight
def gather_flight_features(flight_info: Dict[str, Any]) -> Dict[str, Any]:

    airline = flight_info["airline"]
    departure_airport = flight_info["departure_airport"]
    arrival_airport = flight_info["arrival_airport"]
    scheduled_departure = flight_info["scheduled_departure"]
    scheduled_arrival = flight_info["scheduled_arrival"]
    flight_number = flight_info.get("flight_number", "Unknown")
    
    # Get day of week and hour for the departure
    if isinstance(scheduled_departure, str):
        scheduled_departure = datetime.fromisoformat(scheduled_departure.replace('Z', '+00:00'))
    
    day_of_week = scheduled_departure.weekday()  # 0-6, Monday is 0
    departure_hour = scheduled_departure.hour
    
    # Gather all features
    departure_weather = get_weather_data(departure_airport, scheduled_departure)
    arrival_weather = get_weather_data(arrival_airport, scheduled_arrival)
    
    departure_congestion = get_airport_congestion(departure_airport, scheduled_departure)
    arrival_congestion = get_airport_congestion(arrival_airport, scheduled_arrival)
    
    aircraft_info = get_aircraft_info(airline, flight_number)
    
    historical_performance = get_historical_performance(
        airline, departure_airport, arrival_airport, day_of_week, departure_hour
    )
    
    faa_restrictions = get_faa_restrictions(departure_airport, scheduled_departure)
    faa_restrictions.extend(get_faa_restrictions(arrival_airport, scheduled_arrival))
    
    holiday = is_holiday(scheduled_departure.date())
    
    # Combine into a features dictionary
    return {
        "departure_weather": departure_weather,
        "arrival_weather": arrival_weather,
        "departure_congestion": departure_congestion,
        "arrival_congestion": arrival_congestion,
        "aircraft_info": aircraft_info,
        "historical_airline_delay": historical_performance["airline_delay"],
        "historical_departure_airport_delay": historical_performance["departure_delay"],
        "historical_arrival_airport_delay": historical_performance["arrival_delay"],
        "historical_route_delay": historical_performance["route_delay"],
        "faa_restrictions": faa_restrictions,
        "is_holiday": holiday
    } 