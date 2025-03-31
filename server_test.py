from fastapi import FastAPI, Depends, HTTPException, status
from typing import List, Dict, Any
import uvicorn
from sqlalchemy.orm import Session
import models
import schemas
from database import engine, get_db
from fastapi.middleware.cors import CORSMiddleware
import ml_model

# Create tables if they don't exist
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Flight API", description="API for airport operations dashboard")

# Global variables to store the model and preprocessor
model_classifier = None
model_regressor = None
model_preprocessor = None

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
    global model_classifier, model_regressor, model_preprocessor
    try:
        # Get a new database session
        db = next(get_db())
        try:
            # Initialize the model
            print("Attempting to initialize flight delay prediction model...")
            model_classifier, model_regressor, model_preprocessor = ml_model.initialize_model(db)
            if model_classifier is None:
                print("WARNING: Could not initialize the flight delay prediction model.")
            else:
                print("Flight delay prediction model initialized successfully.")
        finally:
            db.close()
    except Exception as e:
        print(f"ERROR initializing model: {str(e)}")
        print("Server will continue without ML model functionality.")

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
async def predict_flight_delay(request: schemas.DelayPredictionRequest):
    """
    Predict flight delay probability and estimated delay time.
    """
    global model_classifier, model_regressor, model_preprocessor
    
    if model_classifier is None or model_preprocessor is None:
        # Return a default response if the model is not available
        return schemas.DelayPredictionResponse(
            flight_number=f"PRED-{request.airline}-{request.departure_airport}-{request.arrival_airport}",
            delay_probability=0.0,
            estimated_delay_minutes=0,
            prediction_confidence=0.0
        )
    
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
    
    try:
        # Make the prediction
        prediction = ml_model.predict_delay(flight_info, model_classifier, model_regressor, model_preprocessor)
        
        # Return the prediction
        return schemas.DelayPredictionResponse(
            flight_number=prediction["flight_number"],
            delay_probability=prediction["delay_probability"],
            estimated_delay_minutes=prediction["estimated_delay_minutes"],
            prediction_confidence=prediction["prediction_confidence"]
        )
    except Exception as e:
        # If an error occurs during prediction, log it and return a default response
        print(f"Error during prediction: {str(e)}")
        return schemas.DelayPredictionResponse(
            flight_number=f"PRED-{request.airline}-{request.departure_airport}-{request.arrival_airport}",
            delay_probability=0.0,
            estimated_delay_minutes=0,
            prediction_confidence=0.0
        )

@app.post("/train-model", tags=["ML Model"])
async def train_delay_model(db: Session = Depends(get_db)):
    """
    Manually trigger training of the flight delay prediction model.
    """
    global model_classifier, model_regressor, model_preprocessor
    
    try:
        model_classifier, model_regressor, model_preprocessor = ml_model.train_model(db)
        if model_classifier is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Could not train the model. Insufficient data."
            )
        return {"status": "success", "message": "Model successfully trained"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error training model: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run("server_test:app", host="127.0.0.1", port=8000, reload=True) 