from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional, List, Dict, Any, Union

class WeatherConditionBase(BaseModel):
    airport_code: str
    temperature: Optional[float] = None
    wind_speed: Optional[float] = None
    precipitation: Optional[float] = None
    visibility: Optional[float] = None
    storm_indicator: Optional[bool] = None

class WeatherCreate(WeatherConditionBase):
    flight_id: int
    timestamp: datetime
    is_departure: bool

class WeatherCondition(WeatherConditionBase):
    id: int
    flight_id: int
    timestamp: datetime
    is_departure: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)

class AircraftInfo(BaseModel):
    aircraft_type: Optional[str] = None
    aircraft_age: Optional[float] = None

class FlightBase(BaseModel):
    flight_number: str
    airline: str
    departure_airport: str
    arrival_airport: str
    scheduled_departure: datetime
    scheduled_arrival: datetime
    status: str
    gate: Optional[str] = None
    aircraft_type: Optional[str] = None
    aircraft_age: Optional[float] = None
    previous_flight_id: Optional[int] = None
    previous_flight_delay: Optional[int] = None

class FlightCreate(FlightBase):
    pass

class Flight(FlightBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    weather_conditions: Optional[List[WeatherCondition]] = None

    model_config = ConfigDict(from_attributes=True)

class DelayPredictionRequest(BaseModel):
    # Basic flight information (existing features)
    airline: str
    departure_airport: str
    arrival_airport: str
    scheduled_departure: datetime
    scheduled_arrival: datetime
    day_of_week: Optional[int] = None  # 0 for Monday, 6 for Sunday
    month: Optional[int] = None  # 1-12 for January-December
    
    # New features
    # Weather conditions
    departure_weather: Optional[WeatherConditionBase] = None
    arrival_weather: Optional[WeatherConditionBase] = None
    
    # Airport congestion
    departure_congestion: Optional[Dict[str, Union[int, float]]] = None  # {"departures_count": int, "arrivals_count": int, "congestion_level": float}
    arrival_congestion: Optional[Dict[str, Union[int, float]]] = None
    
    # Aircraft specific features
    aircraft_info: Optional[AircraftInfo] = None
    
    # Previous leg delay
    previous_flight_delay: Optional[int] = None
    
    # Holiday and special events
    is_holiday: Optional[bool] = None
    special_event: Optional[str] = None
    
    # Air traffic restrictions
    faa_restrictions: Optional[List[Dict[str, Any]]] = None
    
    # Historical performance
    historical_airline_delay: Optional[float] = None
    historical_departure_airport_delay: Optional[float] = None
    historical_arrival_airport_delay: Optional[float] = None
    historical_route_delay: Optional[float] = None

class DelayPredictionResponse(BaseModel):
    flight_number: str
    delay_probability: float
    estimated_delay_minutes: int
    prediction_confidence: float
    feature_importances: Optional[Dict[str, float]] = None  # Which features contributed most to prediction

class HistoricalPerformanceBase(BaseModel):
    airline: Optional[str] = None
    departure_airport: Optional[str] = None
    arrival_airport: Optional[str] = None
    month: Optional[int] = None
    day_of_week: Optional[int] = None
    hour_of_day: Optional[int] = None
    avg_delay: float
    delay_frequency: float
    sample_size: int

class HistoricalPerformanceCreate(HistoricalPerformanceBase):
    pass

class HistoricalPerformance(HistoricalPerformanceBase):
    id: int
    last_updated: datetime
    
    model_config = ConfigDict(from_attributes=True) 