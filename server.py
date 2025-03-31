from fastapi import FastAPI, Depends, HTTPException, status
from typing import List, Dict, Any
import uvicorn
from sqlalchemy.orm import Session
import models
import schemas
from database import engine, get_db
from fastapi.middleware.cors import CORSMiddleware
import ml_model
import data_sources

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Flight API", description="API for airport operations dashboard")

# Global variables to store the model and preprocessor
classifier = None
regressor = None
preprocessor = None

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """
    Initialize the machine learning model during server startup.
    """
    global classifier, regressor, preprocessor
    # Get a new database session
    db = next(get_db())
    try:
        # Initialize the model
        classifier, regressor, preprocessor = ml_model.initialize_model(db)
        if classifier is None or regressor is None or preprocessor is None:
            print("WARNING: Could not initialize the flight delay prediction model components.")
    finally:
        db.close()

@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify the API is working properly.
    Returns a simple status: ok message.
    """
    return {"status": "ok"}

@app.get("/flights", response_model=List[schemas.Flight], tags=["Flights"])
async def get_flights(db: Session = Depends(get_db)):
    """
    Retrieve all flights from the database.
    """
    flights = db.query(models.Flight).all()
    return flights

@app.get("/flights/{flight_id}", response_model=schemas.Flight, tags=["Flights"])
async def get_flight(flight_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a specific flight by ID.
    """
    flight = db.query(models.Flight).filter(models.Flight.id == flight_id).first()
    if flight is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, 
                            detail=f"Flight with ID {flight_id} not found")
    return flight

@app.post("/flights", response_model=schemas.Flight, status_code=status.HTTP_201_CREATED, tags=["Flights"])
async def create_flight(flight: schemas.FlightCreate, db: Session = Depends(get_db)):
    """
    Create a new flight.
    """
    db_flight = models.Flight(**flight.model_dump())
    db.add(db_flight)
    db.commit()
    db.refresh(db_flight)
    return db_flight

@app.post("/predict-delay", response_model=schemas.DelayPredictionResponse, tags=["Predictions"])
async def predict_flight_delay(request: schemas.DelayPredictionRequest, db: Session = Depends(get_db)):
    """
    Predict flight delay probability and estimated delay time.
    Now supports the enhanced feature set including weather, congestion, etc.
    """
    global classifier, regressor, preprocessor
    
    # Create a flight info dictionary from the request
    flight_info = {
        "flight_number": f"PRED-{request.airline}-{request.departure_airport}-{request.arrival_airport}",
        "airline": request.airline,
        "departure_airport": request.departure_airport,
        "arrival_airport": request.arrival_airport,
        "scheduled_departure": request.scheduled_departure,
        "scheduled_arrival": request.scheduled_arrival,
        "day_of_week": request.day_of_week,
        "month": request.month
    }
    
    # If additional features are not provided in the request, try to fetch them
    if (not request.departure_weather or not request.arrival_weather or 
        not request.departure_congestion or not request.arrival_congestion or
        not request.aircraft_info or request.historical_airline_delay is None):
        
        # Gather external data for features not provided in the request
        try:
            external_features = data_sources.gather_flight_features(flight_info)
            
            # Only use external features for fields not provided in the request
            if not request.departure_weather:
                flight_info["departure_weather"] = external_features["departure_weather"]
            else:
                flight_info["departure_weather"] = request.departure_weather.model_dump()
                
            if not request.arrival_weather:
                flight_info["arrival_weather"] = external_features["arrival_weather"]
            else:
                flight_info["arrival_weather"] = request.arrival_weather.model_dump()
                
            if not request.departure_congestion:
                flight_info["departure_congestion"] = external_features["departure_congestion"]
            else:
                flight_info["departure_congestion"] = request.departure_congestion
                
            if not request.arrival_congestion:
                flight_info["arrival_congestion"] = external_features["arrival_congestion"]
            else:
                flight_info["arrival_congestion"] = request.arrival_congestion
                
            if not request.aircraft_info:
                flight_info["aircraft_info"] = external_features["aircraft_info"]
            else:
                flight_info["aircraft_info"] = request.aircraft_info.model_dump()
                
            if request.historical_airline_delay is None:
                flight_info["historical_airline_delay"] = external_features["historical_airline_delay"]
            else:
                flight_info["historical_airline_delay"] = request.historical_airline_delay
                
            if request.historical_departure_airport_delay is None:
                flight_info["historical_departure_airport_delay"] = external_features["historical_departure_airport_delay"]
            else:
                flight_info["historical_departure_airport_delay"] = request.historical_departure_airport_delay
                
            if request.historical_arrival_airport_delay is None:
                flight_info["historical_arrival_airport_delay"] = external_features["historical_arrival_airport_delay"]
            else:
                flight_info["historical_arrival_airport_delay"] = request.historical_arrival_airport_delay
                
            if request.historical_route_delay is None:
                flight_info["historical_route_delay"] = external_features["historical_route_delay"]
            else:
                flight_info["historical_route_delay"] = request.historical_route_delay
                
            if request.is_holiday is None:
                flight_info["is_holiday"] = external_features["is_holiday"]
            else:
                flight_info["is_holiday"] = request.is_holiday
                
            if not request.faa_restrictions:
                flight_info["faa_restrictions"] = external_features["faa_restrictions"]
            else:
                flight_info["faa_restrictions"] = request.faa_restrictions
                
            if request.previous_flight_delay is None and "previous_flight_delay" in external_features:
                flight_info["previous_flight_delay"] = external_features["previous_flight_delay"]
            else:
                flight_info["previous_flight_delay"] = request.previous_flight_delay
                
        except Exception as e:
            print(f"Error fetching external data: {str(e)}")
            # Continue with whatever data we have
    else:
        # Use the data provided in the request
        if request.departure_weather:
            flight_info["departure_weather"] = request.departure_weather.model_dump()
        if request.arrival_weather:
            flight_info["arrival_weather"] = request.arrival_weather.model_dump()
        if request.departure_congestion:
            flight_info["departure_congestion"] = request.departure_congestion
        if request.arrival_congestion:
            flight_info["arrival_congestion"] = request.arrival_congestion
        if request.aircraft_info:
            flight_info["aircraft_info"] = request.aircraft_info.model_dump()
        if request.previous_flight_delay is not None:
            flight_info["previous_flight_delay"] = request.previous_flight_delay
        if request.is_holiday is not None:
            flight_info["is_holiday"] = request.is_holiday
        if request.special_event:
            flight_info["special_event"] = request.special_event
        if request.faa_restrictions:
            flight_info["faa_restrictions"] = request.faa_restrictions
        if request.historical_airline_delay is not None:
            flight_info["historical_airline_delay"] = request.historical_airline_delay
        if request.historical_departure_airport_delay is not None:
            flight_info["historical_departure_airport_delay"] = request.historical_departure_airport_delay
        if request.historical_arrival_airport_delay is not None:
            flight_info["historical_arrival_airport_delay"] = request.historical_arrival_airport_delay
        if request.historical_route_delay is not None:
            flight_info["historical_route_delay"] = request.historical_route_delay
    
    # Make the prediction using the db connection for any additional data lookup
    prediction = ml_model.predict_delay(flight_info, classifier, regressor, preprocessor, db)
    
    # Return the prediction
    return schemas.DelayPredictionResponse(
        flight_number=prediction["flight_number"],
        delay_probability=prediction["delay_probability"],
        estimated_delay_minutes=prediction["estimated_delay_minutes"],
        prediction_confidence=prediction["prediction_confidence"],
        feature_importances=prediction.get("feature_importances", {})
    )

@app.post("/train-model", tags=["ML Model"])
async def train_delay_model(db: Session = Depends(get_db)):
    """
    Manually trigger training of the flight delay prediction model.
    """
    global classifier, regressor, preprocessor
    
    try:
        classifier, regressor, preprocessor = ml_model.train_model(db)
        if classifier is None or regressor is None or preprocessor is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not train the model. Insufficient data."
            )
        return {
            "status": "success", 
            "message": "Model successfully trained",
            "details": {
                "classifier": type(classifier).__name__,
                "regressor": type(regressor).__name__ if regressor else "Not trained (insufficient delay data)",
                "num_features": getattr(preprocessor, 'n_features_in_', 'Unknown')
            }
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training model: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("server:app", host="127.0.0.1", port=8000, reload=True) 