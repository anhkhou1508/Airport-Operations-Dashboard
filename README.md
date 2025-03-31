# Airport Operations Dashboard API

A FastAPI-based API for an airport operations dashboard that provides flight information and delay predictions using machine learning.

## Current Project State

**Important Note**: This project is currently a **proof-of-concept/demonstration** with the following characteristics:

- The ML model is trained on **synthetic/generated data**, not real flight data
- Delay predictions are based on patterns in this synthetic data
- External data features (weather, congestion, etc.) are simulated when not provided
- The system architecture demonstrates how a real-world implementation would work

## Features

- PostgreSQL database with SQLAlchemy ORM
- Health check endpoint (`/health`) to verify the API is functioning
- Flight information endpoints to interact with flight data
- Machine learning model for flight delay prediction using:
  - Basic flight information (airline, airports, times)
  - Weather conditions (temperature, wind, precipitation, visibility)
  - Airport congestion levels
  - Aircraft information (type, age)
  - Historical performance metrics
- Interactive API documentation at `/docs` (powered by Swagger UI)

## Prerequisites

- Python 3.7+
- PostgreSQL server running

## Setup

1. Clone the repository

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Configure the PostgreSQL database:
   - Create a PostgreSQL database named `flights_db`
   - Update the `.env` file with your PostgreSQL connection string if necessary
   - Default configuration is: `postgresql://postgres:postgres@localhost/flights_db`

4. Initialize the database and seed with initial sample data:
```bash
python init_db.py
```

5. Generate additional sample data for training the ML model:
```bash
python generate_sample_data.py
```

6. Run the server:
```bash
python server.py
```

The server will start running at `http://localhost:8000`. The ML model will be initialized automatically on startup using the generated data.

## Testing Features

1. Verify the server is running:
```bash
python check_server.py
```

2. Test the ML prediction model with comprehensive features:
```bash
python test_prediction.py
```
This script sends a prediction request with all available features and analyzes the response, showing which features most influence the prediction.

3. Use the Swagger UI to test endpoints:
   - Open your browser to: `http://localhost:8000/docs`
   - Test the `/predict-delay` endpoint with various inputs
   - Experiment with the `/train-model` endpoint to retrain the model

4. Understanding Predictions with Minimal Input:
   - When only basic flight information is provided (airline, airports, times), the system automatically gathers simulated data for missing features rather than using random values
   - The `data_sources.py` module provides realistic but simulated values for weather, congestion, etc.
   - You can view the feature importances in prediction responses to see which features influenced the prediction

## Endpoints

### Health Check

```
GET /health
```

Returns:
```json
{
  "status": "ok"
}
```

### Get All Flights

```
GET /flights
```

Returns a list of flight data from the database.

### Get Flight by ID

```
GET /flights/{flight_id}
```

Returns a specific flight by ID.

### Create New Flight

```
POST /flights
```

Creates a new flight record in the database.

### Predict Flight Delay

```
POST /predict-delay
```

Predicts the probability of delay and estimated delay time for a flight using our prediction model.

Example basic request (system will provide simulated values for missing features):
```json
{
  "airline": "Delta",
  "departure_airport": "ATL",
  "arrival_airport": "LAX",
  "scheduled_departure": "2023-12-15T13:30:00",
  "scheduled_arrival": "2023-12-15T15:45:00"
}
```

Example comprehensive request with all features:
```json
{
  "airline": "Delta",
  "departure_airport": "ATL",
  "arrival_airport": "LAX",
  "scheduled_departure": "2023-12-15T13:30:00",
  "scheduled_arrival": "2023-12-15T15:45:00",
  "departure_weather": {
    "airport_code": "ATL",
    "temperature": 28.5,
    "wind_speed": 15.3,
    "precipitation": 0.2,
    "visibility": 8.5,
    "storm_indicator": false
  },
  "arrival_weather": {
    "airport_code": "LAX",
    "temperature": 22.1,
    "wind_speed": 8.7,
    "precipitation": 0.0,
    "visibility": 10.0,
    "storm_indicator": false
  },
  "departure_congestion": {
    "departures_count": 45,
    "arrivals_count": 38,
    "congestion_level": 0.75
  },
  "aircraft_info": {
    "aircraft_type": "Boeing 737-800",
    "aircraft_age": 7.5
  },
  "previous_flight_delay": 25,
  "is_holiday": true,
  "historical_airline_delay": 12.5,
  "historical_departure_airport_delay": 15.2,
  "historical_arrival_airport_delay": 8.7,
  "historical_route_delay": 10.3
}
```

Example response:
```json
{
  "flight_number": "PRED-Delta-ATL-LAX",
  "delay_probability": 0.65,
  "estimated_delay_minutes": 45,
  "prediction_confidence": 0.82,
  "feature_importances": {
    "weather_departure_precipitation": 0.15,
    "congestion_departure_congestion_level": 0.12,
    "historical_historical_airline_delay": 0.10,
    "cat_departure_airport": 0.08,
    "weather_arrival_visibility": 0.07
  }
}
```

### Train Model

```
POST /train-model
```

Manually triggers retraining of the flight delay prediction model using the latest data in the database.

## Machine Learning Implementation

The ML component uses:

1. **RandomForest** models for both:
   - Classification (predicting delay probability)
   - Regression (predicting delay duration in minutes)
   
2. **Synthetic training data** with:
   - Basic flight information generated randomly
   - Weather conditions simulated with realistic constraints
   - Congestion patterns that follow typical airport busy periods
   - Random but plausible delay patterns

3. **Feature processing pipeline** that:
   - Handles missing values through imputation
   - Encodes categorical features
   - Scales numerical features appropriately
   - Produces feature importance analysis

## Future Development

To make this project production-ready, consider these enhancements:

1. Replace synthetic data with real historical flight data from:
   - FAA datasets
   - Bureau of Transportation Statistics 
   - Commercial flight data providers

2. Integrate with actual weather APIs and airport data sources

3. Enhance the ML model with:
   - More sophisticated algorithms
   - Hyperparameter optimization
   - Time-series analysis components

4. Add a frontend dashboard to visualize predictions and historical patterns

## Database Schema

The database has tables to support the prediction model:

- `flights`: Core flight information plus aircraft data
- `weather_conditions`: Weather data for departure and arrival airports
- `airport_congestion`: Airport traffic volume and congestion metrics
- `air_traffic_restrictions`: FAA and other regulatory restrictions
- `holiday_events`: Holidays and special events that may affect travel
- `historical_performance`: Performance metrics for airlines, airports, and routes