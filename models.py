from sqlalchemy import Column, Integer, String, DateTime, Enum, Float, Boolean, ForeignKey
from sqlalchemy.sql import func
import enum
from sqlalchemy.orm import relationship
from database import Base

class FlightStatus(enum.Enum):
    ON_TIME = "On Time"
    DELAYED = "Delayed"
    BOARDING = "Boarding"
    DEPARTED = "Departed"
    ARRIVED = "Arrived"
    CANCELLED = "Cancelled"

class Flight(Base):
    __tablename__ = "flights"

    id = Column(Integer, primary_key=True, index=True)
    flight_number = Column(String, unique=True, index=True, nullable=False)
    airline = Column(String, nullable=False)
    departure_airport = Column(String, nullable=False)
    arrival_airport = Column(String, nullable=False)
    scheduled_departure = Column(DateTime, nullable=False)
    scheduled_arrival = Column(DateTime, nullable=False)
    status = Column(String, nullable=False, default=FlightStatus.ON_TIME.value)
    gate = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Aircraft information
    aircraft_type = Column(String, nullable=True)
    aircraft_age = Column(Float, nullable=True)
    
    # Previous leg delay information
    previous_flight_id = Column(Integer, ForeignKey("flights.id"), nullable=True)
    previous_flight = relationship("Flight", remote_side=[id], backref="next_flights", uselist=False)
    previous_flight_delay = Column(Integer, nullable=True)  # Delay in minutes
    
    # Relationships with other tables
    weather_conditions = relationship("WeatherCondition", back_populates="flight")
    
class WeatherCondition(Base):
    __tablename__ = "weather_conditions"
    
    id = Column(Integer, primary_key=True, index=True)
    flight_id = Column(Integer, ForeignKey("flights.id"))
    airport_code = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    temperature = Column(Float, nullable=True)  # in Celsius
    wind_speed = Column(Float, nullable=True)   # in km/h
    precipitation = Column(Float, nullable=True) # in mm
    visibility = Column(Float, nullable=True)   # in km
    storm_indicator = Column(Boolean, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Whether this is departure or arrival airport weather
    is_departure = Column(Boolean, default=True)
    
    # Relationship with flight
    flight = relationship("Flight", back_populates="weather_conditions")

class AirportCongestion(Base):
    __tablename__ = "airport_congestion"
    
    id = Column(Integer, primary_key=True, index=True)
    airport_code = Column(String, nullable=False)
    timestamp = Column(DateTime, nullable=False)
    departures_count = Column(Integer, default=0)
    arrivals_count = Column(Integer, default=0)
    congestion_level = Column(Float, default=0.0)  # 0-1 scale
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class AirTrafficRestriction(Base):
    __tablename__ = "air_traffic_restrictions"
    
    id = Column(Integer, primary_key=True, index=True)
    airport_code = Column(String, nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    restriction_type = Column(String, nullable=False)  # e.g., "Ground Stop", "Ground Delay"
    reason = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class HolidayEvent(Base):
    __tablename__ = "holiday_events"
    
    id = Column(Integer, primary_key=True, index=True)
    date = Column(DateTime, nullable=False)
    name = Column(String, nullable=False)
    is_holiday = Column(Boolean, default=True)
    country = Column(String, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class HistoricalPerformance(Base):
    __tablename__ = "historical_performance"
    
    id = Column(Integer, primary_key=True, index=True)
    airline = Column(String, nullable=True)
    departure_airport = Column(String, nullable=True)
    arrival_airport = Column(String, nullable=True)
    month = Column(Integer, nullable=True)
    day_of_week = Column(Integer, nullable=True)
    hour_of_day = Column(Integer, nullable=True)
    avg_delay = Column(Float, default=0.0)  # Average delay in minutes
    delay_frequency = Column(Float, default=0.0)  # Percentage of flights delayed
    sample_size = Column(Integer, default=0)  # Number of flights in the sample
    last_updated = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now()) 