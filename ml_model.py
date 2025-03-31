import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import os
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import func, and_, or_
import models
from database import engine, get_db

# File paths for saving and loading the model
MODEL_DIR = "ml_models"
CLASSIFIER_PATH = os.path.join(MODEL_DIR, "flight_delay_classifier.joblib")
REGRESSOR_PATH = os.path.join(MODEL_DIR, "flight_delay_regressor.joblib")
PREPROCESSOR_PATH = os.path.join(MODEL_DIR, "preprocessor.joblib")

# Ensure the model directory exists
os.makedirs(MODEL_DIR, exist_ok=True)

# Custom transformer for extracting holiday features
class HolidayTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, db_session=None):
        self.db_session = db_session
        self.holidays = {}
        if db_session:
            self._load_holidays()
    
    def _load_holidays(self):
        # Load holidays from database
        holiday_records = self.db_session.query(models.HolidayEvent).all()
        for record in holiday_records:
            date_key = record.date.strftime('%Y-%m-%d')
            self.holidays[date_key] = record.name
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Check if date is in holidays dictionary
        is_holiday = np.zeros((X.shape[0], 1))
        
        for i, date in enumerate(X['scheduled_departure'].dt.strftime('%Y-%m-%d')):
            if date in self.holidays:
                is_holiday[i, 0] = 1
                
        return is_holiday

def extract_features(flight_data, db=None):
    """
    Extract relevant features from flight data for the model.
    Now includes additional features like weather, congestion, aircraft info, etc.
    """
    # Extract time-based features (existing features)
    flight_data['scheduled_departure_hour'] = flight_data['scheduled_departure'].dt.hour
    flight_data['scheduled_departure_day'] = flight_data['scheduled_departure'].dt.day
    flight_data['day_of_week'] = flight_data['scheduled_departure'].dt.dayofweek  # 0 = Monday, 6 = Sunday
    flight_data['month'] = flight_data['scheduled_departure'].dt.month
    
    # Calculate flight duration in minutes (existing feature)
    flight_data['planned_duration'] = ((flight_data['scheduled_arrival'] - 
                                      flight_data['scheduled_departure']).dt.total_seconds() / 60)
    
    # Add new features
    
    # 1. Aircraft-specific features
    flight_data['aircraft_type_available'] = flight_data['aircraft_type'].notna().astype(int)
    flight_data['aircraft_age'] = flight_data.get('aircraft_age', np.nan)
    
    # 2. Previous leg delay
    flight_data['has_previous_leg'] = flight_data['previous_flight_id'].notna().astype(int)
    flight_data['previous_leg_delay'] = flight_data.get('previous_flight_delay', 0)
    
    # 3. Holiday indicator
    if db:
        # Get holiday data from database
        flight_dates = flight_data['scheduled_departure'].dt.strftime('%Y-%m-%d').unique()
        holidays = db.query(models.HolidayEvent).filter(
            func.date(models.HolidayEvent.date).in_(flight_dates)
        ).all()
        
        # Create dictionary mapping dates to holiday status
        holiday_dict = {h.date.strftime('%Y-%m-%d'): 1 for h in holidays}
        
        # Map to flight_data
        flight_data['is_holiday'] = flight_data['scheduled_departure'].dt.strftime('%Y-%m-%d').map(
            holiday_dict).fillna(0).astype(int)
    else:
        flight_data['is_holiday'] = 0
    
    # 4. Peak travel periods (simplified - weekends and holiday seasons)
    flight_data['is_weekend'] = (flight_data['day_of_week'] >= 5).astype(int)
    flight_data['is_summer'] = ((flight_data['month'] >= 6) & (flight_data['month'] <= 8)).astype(int)
    flight_data['is_winter_holiday'] = ((flight_data['month'] == 12) | (flight_data['month'] == 1)).astype(int)
    
    # 5. Weather conditions
    # Add placeholders if weather data not available
    weather_features = ['departure_temperature', 'departure_wind_speed', 'departure_precipitation', 
                        'departure_visibility', 'departure_storm', 
                        'arrival_temperature', 'arrival_wind_speed', 'arrival_precipitation', 
                        'arrival_visibility', 'arrival_storm']
    
    # Add weather data if available in the dataframe or from related tables
    if 'departure_weather' in flight_data.columns and not flight_data['departure_weather'].isna().all():
        # If weather data is included as a dict/object in a column
        flight_data['departure_temperature'] = flight_data['departure_weather'].apply(
            lambda x: x.get('temperature') if x else None)
        flight_data['departure_wind_speed'] = flight_data['departure_weather'].apply(
            lambda x: x.get('wind_speed') if x else None)
        flight_data['departure_precipitation'] = flight_data['departure_weather'].apply(
            lambda x: x.get('precipitation') if x else None)
        flight_data['departure_visibility'] = flight_data['departure_weather'].apply(
            lambda x: x.get('visibility') if x else None)
        flight_data['departure_storm'] = flight_data['departure_weather'].apply(
            lambda x: x.get('storm_indicator', 0) if x else 0)
        
        flight_data['arrival_temperature'] = flight_data['arrival_weather'].apply(
            lambda x: x.get('temperature') if x else None)
        flight_data['arrival_wind_speed'] = flight_data['arrival_weather'].apply(
            lambda x: x.get('wind_speed') if x else None)
        flight_data['arrival_precipitation'] = flight_data['arrival_weather'].apply(
            lambda x: x.get('precipitation') if x else None)
        flight_data['arrival_visibility'] = flight_data['arrival_weather'].apply(
            lambda x: x.get('visibility') if x else None)
        flight_data['arrival_storm'] = flight_data['arrival_weather'].apply(
            lambda x: x.get('storm_indicator', 0) if x else 0)
    elif db and 'id' in flight_data.columns:
        # Query the weather data from database for these flights
        flight_ids = flight_data['id'].tolist()
        weather_records = db.query(models.WeatherCondition).filter(
            models.WeatherCondition.flight_id.in_(flight_ids)
        ).all()
        
        # Group by flight_id and is_departure
        departure_weather = {}
        arrival_weather = {}
        
        for record in weather_records:
            if record.is_departure:
                departure_weather[record.flight_id] = {
                    'temperature': record.temperature,
                    'wind_speed': record.wind_speed,
                    'precipitation': record.precipitation,
                    'visibility': record.visibility,
                    'storm': record.storm_indicator
                }
            else:
                arrival_weather[record.flight_id] = {
                    'temperature': record.temperature,
                    'wind_speed': record.wind_speed,
                    'precipitation': record.precipitation,
                    'visibility': record.visibility,
                    'storm': record.storm_indicator
                }
        
        # Add to flight_data
        for feature in ['temperature', 'wind_speed', 'precipitation', 'visibility', 'storm']:
            flight_data[f'departure_{feature}'] = flight_data['id'].map(
                lambda x: departure_weather.get(x, {}).get(feature, None))
            flight_data[f'arrival_{feature}'] = flight_data['id'].map(
                lambda x: arrival_weather.get(x, {}).get(feature, None))
    else:
        # Initialize with NaN (will be imputed during preprocessing)
        for feature in weather_features:
            flight_data[feature] = np.nan
    
    # 6. Airport congestion data
    congestion_features = ['departure_congestion_level', 'arrival_congestion_level',
                         'departure_flights_count', 'arrival_flights_count']
    
    if 'departure_congestion' in flight_data.columns and not flight_data['departure_congestion'].isna().all():
        # If congestion data is included as a dict/object
        flight_data['departure_congestion_level'] = flight_data['departure_congestion'].apply(
            lambda x: x.get('congestion_level') if x else None)
        flight_data['departure_flights_count'] = flight_data['departure_congestion'].apply(
            lambda x: x.get('departures_count', 0) + x.get('arrivals_count', 0) if x else 0)
        
        flight_data['arrival_congestion_level'] = flight_data['arrival_congestion'].apply(
            lambda x: x.get('congestion_level') if x else None)
        flight_data['arrival_flights_count'] = flight_data['arrival_congestion'].apply(
            lambda x: x.get('departures_count', 0) + x.get('arrivals_count', 0) if x else 0)
    elif db:
        # Query congestion data from the database
        # We need to match by airport and time
        congestion_data = {}
        unique_airports = list(set(flight_data['departure_airport'].tolist() + 
                                 flight_data['arrival_airport'].tolist()))
        
        # For each flight, get the closest congestion record to the departure/arrival time
        for idx, flight in flight_data.iterrows():
            # Departure congestion
            dep_airport = flight['departure_airport']
            dep_time = flight['scheduled_departure']
            
            dep_congestion = db.query(models.AirportCongestion).filter(
                models.AirportCongestion.airport_code == dep_airport,
                models.AirportCongestion.timestamp <= dep_time
            ).order_by(models.AirportCongestion.timestamp.desc()).first()
            
            if dep_congestion:
                flight_data.at[idx, 'departure_congestion_level'] = dep_congestion.congestion_level
                flight_data.at[idx, 'departure_flights_count'] = (
                    dep_congestion.departures_count + dep_congestion.arrivals_count
                )
            
            # Arrival congestion
            arr_airport = flight['arrival_airport']
            arr_time = flight['scheduled_arrival']
            
            arr_congestion = db.query(models.AirportCongestion).filter(
                models.AirportCongestion.airport_code == arr_airport,
                models.AirportCongestion.timestamp <= arr_time
            ).order_by(models.AirportCongestion.timestamp.desc()).first()
            
            if arr_congestion:
                flight_data.at[idx, 'arrival_congestion_level'] = arr_congestion.congestion_level
                flight_data.at[idx, 'arrival_flights_count'] = (
                    arr_congestion.departures_count + arr_congestion.arrivals_count
                )
    else:
        # Initialize with NaN (will be imputed during preprocessing)
        for feature in congestion_features:
            flight_data[feature] = np.nan
    
    # 7. Historical performance metrics
    performance_features = ['historical_airline_delay', 'historical_departure_delay', 
                          'historical_arrival_delay', 'historical_route_delay']
    
    if all(f in flight_data.columns for f in performance_features):
        # Performance data is already included
        pass
    elif db:
        # Query historical performance from database
        for idx, flight in flight_data.iterrows():
            # Airline performance
            airline_perf = db.query(models.HistoricalPerformance).filter(
                models.HistoricalPerformance.airline == flight['airline']
            ).first()
            if airline_perf:
                flight_data.at[idx, 'historical_airline_delay'] = airline_perf.avg_delay
            
            # Departure airport performance
            dep_perf = db.query(models.HistoricalPerformance).filter(
                models.HistoricalPerformance.departure_airport == flight['departure_airport']
            ).first()
            if dep_perf:
                flight_data.at[idx, 'historical_departure_delay'] = dep_perf.avg_delay
            
            # Arrival airport performance
            arr_perf = db.query(models.HistoricalPerformance).filter(
                models.HistoricalPerformance.arrival_airport == flight['arrival_airport']
            ).first()
            if arr_perf:
                flight_data.at[idx, 'historical_arrival_delay'] = arr_perf.avg_delay
            
            # Route performance (specific airline, departure, and arrival)
            route_perf = db.query(models.HistoricalPerformance).filter(
                models.HistoricalPerformance.airline == flight['airline'],
                models.HistoricalPerformance.departure_airport == flight['departure_airport'],
                models.HistoricalPerformance.arrival_airport == flight['arrival_airport']
            ).first()
            if route_perf:
                flight_data.at[idx, 'historical_route_delay'] = route_perf.avg_delay
    else:
        # Initialize with 0 (neutral value for historical delays)
        for feature in performance_features:
            flight_data[feature] = 0
    
    # 8. FAA restrictions
    flight_data['has_faa_restriction'] = 0
    if db:
        # Check if there are any active FAA restrictions for each flight
        for idx, flight in flight_data.iterrows():
            # Check departure restrictions
            dep_restrictions = db.query(models.AirTrafficRestriction).filter(
                models.AirTrafficRestriction.airport_code == flight['departure_airport'],
                models.AirTrafficRestriction.start_time <= flight['scheduled_departure'],
                models.AirTrafficRestriction.end_time >= flight['scheduled_departure']
            ).first()
            
            # Check arrival restrictions
            arr_restrictions = db.query(models.AirTrafficRestriction).filter(
                models.AirTrafficRestriction.airport_code == flight['arrival_airport'],
                models.AirTrafficRestriction.start_time <= flight['scheduled_arrival'],
                models.AirTrafficRestriction.end_time >= flight['scheduled_arrival']
            ).first()
            
            if dep_restrictions or arr_restrictions:
                flight_data.at[idx, 'has_faa_restriction'] = 1
    
    # Create features dataframe with all the relevant features
    features = flight_data[['airline', 'departure_airport', 'arrival_airport', 
                           'scheduled_departure_hour', 'day_of_week', 'month', 'planned_duration',
                           # Aircraft features
                           'aircraft_type', 'aircraft_age', 'aircraft_type_available',
                           # Previous leg
                           'has_previous_leg', 'previous_leg_delay',
                           # Time-related features
                           'is_weekend', 'is_holiday', 'is_summer', 'is_winter_holiday',
                           # Weather at departure
                           'departure_temperature', 'departure_wind_speed', 'departure_precipitation', 
                           'departure_visibility', 'departure_storm',
                           # Weather at arrival
                           'arrival_temperature', 'arrival_wind_speed', 'arrival_precipitation', 
                           'arrival_visibility', 'arrival_storm',
                           # Airport congestion
                           'departure_congestion_level', 'arrival_congestion_level',
                           'departure_flights_count', 'arrival_flights_count',
                           # Historical performance
                           'historical_airline_delay', 'historical_departure_delay', 
                           'historical_arrival_delay', 'historical_route_delay',
                           # FAA restrictions
                           'has_faa_restriction']]
    
    # Target: Whether the flight was delayed (if status contains "Delayed")
    is_delayed = flight_data['status'].str.contains('Delayed').astype(int)
    
    # Calculate delay duration (for regression task)
    delayed_flights = flight_data[flight_data['status'].str.contains('Delayed')]
    delay_durations = pd.Series(index=flight_data.index, data=0)  # Default 0 for non-delayed flights
    
    if not delayed_flights.empty:
        for idx, flight in delayed_flights.iterrows():
            # Estimate delay as 15-180 minutes (placeholder for actual delay data)
            delay_durations.loc[idx] = np.random.randint(15, 180)
    
    return features, is_delayed, delay_durations

def create_preprocessor():
    """
    Create a preprocessor for the flight data.
    Handles all the new features added to the model.
    """
    # Define the preprocessing for the categorical features
    categorical_features = ['airline', 'departure_airport', 'arrival_airport', 'aircraft_type']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='UNKNOWN')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Define the preprocessing for the basic numerical features
    numerical_features = ['scheduled_departure_hour', 'day_of_week', 'month', 'planned_duration',
                         'aircraft_age', 'previous_leg_delay',
                         'is_weekend', 'is_holiday', 'is_summer', 'is_winter_holiday',
                         'has_previous_leg', 'aircraft_type_available', 'has_faa_restriction']
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Special preprocessing for weather features that may have more missing values
    weather_features = ['departure_temperature', 'departure_wind_speed', 'departure_precipitation', 
                      'departure_visibility', 'departure_storm',
                      'arrival_temperature', 'arrival_wind_speed', 'arrival_precipitation', 
                      'arrival_visibility', 'arrival_storm']
    weather_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())  # Use RobustScaler for weather which may have outliers
    ])
    
    # Preprocessing for congestion features
    congestion_features = ['departure_congestion_level', 'arrival_congestion_level',
                         'departure_flights_count', 'arrival_flights_count']
    congestion_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Preprocessing for historical performance features
    historical_features = ['historical_airline_delay', 'historical_departure_delay', 
                         'historical_arrival_delay', 'historical_route_delay']
    historical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value=0)),  # Use 0 for missing historical data
        ('scaler', StandardScaler())
    ])
    
    # Combine all the preprocessors
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, categorical_features),
            ('num', numerical_transformer, numerical_features),
            ('weather', weather_transformer, weather_features),
            ('congestion', congestion_transformer, congestion_features),
            ('historical', historical_transformer, historical_features)
        ])
    
    return preprocessor

def train_model(db: Session):
    """
    Train the flight delay prediction model using data from the database.
    Now incorporates all the new features like weather, congestion, etc.
    """
    # Get flight data from the database
    flight_records = db.query(models.Flight).all()
    
    # Convert to DataFrame
    flight_data = pd.DataFrame([{
        'id': flight.id,
        'flight_number': flight.flight_number,
        'airline': flight.airline,
        'departure_airport': flight.departure_airport,
        'arrival_airport': flight.arrival_airport,
        'scheduled_departure': flight.scheduled_departure,
        'scheduled_arrival': flight.scheduled_arrival,
        'status': flight.status,
        'aircraft_type': flight.aircraft_type,
        'aircraft_age': flight.aircraft_age,
        'previous_flight_id': flight.previous_flight_id,
        'previous_flight_delay': flight.previous_flight_delay
    } for flight in flight_records])
    
    # If no data, return early
    if len(flight_data) < 10:
        print("Not enough flight data to train the model. Need at least 10 flights.")
        return None, None, None
    
    # Extract features
    features, is_delayed, delay_durations = extract_features(flight_data, db)
    
    # Split the data for classification task (predicting delay probability)
    X_train, X_test, y_train, y_test = train_test_split(features, is_delayed, test_size=0.2, random_state=42)
    
    # Create and fit the preprocessor
    preprocessor = create_preprocessor()
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    # Save the preprocessor
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    
    # Train classifier model (for delay probability)
    classifier = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        class_weight='balanced',
        random_state=42
    )
    classifier.fit(X_train_processed, y_train)
    
    # Evaluate classifier
    classifier_score = classifier.score(X_test_processed, y_test)
    print(f"Classifier accuracy: {classifier_score:.4f}")
    
    # Train regressor model (for delay duration)
    regressor = RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features='sqrt',
        bootstrap=True,
        random_state=42
    )
    
    # Only fit if we have actual delayed flights
    if sum(is_delayed) > 0:
        # Filter to only delayed flights for regression
        delayed_idx = is_delayed[is_delayed == 1].index
        X_delayed = preprocessor.transform(features.loc[delayed_idx])
        y_delayed = delay_durations.loc[delayed_idx]
        regressor.fit(X_delayed, y_delayed)
        
        # Calculate and print feature importances
        if hasattr(classifier, 'feature_importances_'):
            importances = classifier.feature_importances_
            indices = np.argsort(importances)[::-1]
            print("Top 10 most important features for delay prediction:")
            for i in range(min(10, len(indices))):
                print(f"  {i+1}. Feature index {indices[i]}: {importances[indices[i]]:.4f}")
    else:
        print("No delayed flights in the dataset to train the regressor.")
    
    # Save the models
    joblib.dump(classifier, CLASSIFIER_PATH)
    joblib.dump(regressor, REGRESSOR_PATH)
    
    return classifier, regressor, preprocessor

def load_model():
    """
    Load the trained model and preprocessor if they exist.
    Returns the classifier, regressor, and preprocessor.
    """
    try:
        if os.path.exists(CLASSIFIER_PATH) and os.path.exists(REGRESSOR_PATH) and os.path.exists(PREPROCESSOR_PATH):
            classifier = joblib.load(CLASSIFIER_PATH)
            regressor = joblib.load(REGRESSOR_PATH)
            preprocessor = joblib.load(PREPROCESSOR_PATH)
            print("Successfully loaded existing model, regressor, and preprocessor.")
            return classifier, regressor, preprocessor
        else:
            print("One or more model files not found. Need to train new models.")
            return None, None, None
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None, None, None

def predict_delay(flight_info, classifier=None, regressor=None, preprocessor=None, db=None):
    """
    Predict flight delay probability and estimated delay time.
    Now supports the full feature set including weather, congestion, etc.
    """
    # Load model and preprocessor if not provided
    if classifier is None or regressor is None or preprocessor is None:
        classifier, regressor, preprocessor = load_model()
        
    # If models or preprocessor is still None, we can't make a prediction
    if classifier is None or preprocessor is None:
        return {
            "flight_number": flight_info.get("flight_number", "Unknown"),
            "delay_probability": 0.0,
            "estimated_delay_minutes": 0,
            "prediction_confidence": 0.0,
            "feature_importances": {}
        }
    
    # Create a DataFrame with the flight information
    flight_df = pd.DataFrame([{
        'airline': flight_info['airline'],
        'departure_airport': flight_info['departure_airport'],
        'arrival_airport': flight_info['arrival_airport'],
        'scheduled_departure': flight_info['scheduled_departure'],
        'scheduled_arrival': flight_info['scheduled_arrival'],
        'status': flight_info.get('status', 'Unknown')  # Default for prediction purposes
    }])
    
    # Add weather information if available
    if 'departure_weather' in flight_info and flight_info['departure_weather']:
        flight_df['departure_weather'] = [flight_info['departure_weather']]
    if 'arrival_weather' in flight_info and flight_info['arrival_weather']:
        flight_df['arrival_weather'] = [flight_info['arrival_weather']]
    
    # Add congestion information if available
    if 'departure_congestion' in flight_info and flight_info['departure_congestion']:
        flight_df['departure_congestion'] = [flight_info['departure_congestion']]
    if 'arrival_congestion' in flight_info and flight_info['arrival_congestion']:
        flight_df['arrival_congestion'] = [flight_info['arrival_congestion']]
    
    # Add aircraft information if available
    if 'aircraft_info' in flight_info and flight_info['aircraft_info']:
        flight_df['aircraft_type'] = flight_info['aircraft_info'].get('aircraft_type')
        flight_df['aircraft_age'] = flight_info['aircraft_info'].get('aircraft_age')
    
    # Add previous leg delay if available
    if 'previous_flight_delay' in flight_info and flight_info['previous_flight_delay'] is not None:
        flight_df['previous_flight_id'] = 1  # Placeholder ID to indicate we have previous flight info
        flight_df['previous_flight_delay'] = flight_info['previous_flight_delay']
    
    # Add holiday/special event information
    if 'is_holiday' in flight_info and flight_info['is_holiday'] is not None:
        flight_df['is_holiday'] = flight_info['is_holiday']
    
    # Add historical performance if available
    if 'historical_airline_delay' in flight_info and flight_info['historical_airline_delay'] is not None:
        flight_df['historical_airline_delay'] = flight_info['historical_airline_delay']
    if 'historical_departure_airport_delay' in flight_info and flight_info['historical_departure_airport_delay'] is not None:
        flight_df['historical_departure_delay'] = flight_info['historical_departure_airport_delay']
    if 'historical_arrival_airport_delay' in flight_info and flight_info['historical_arrival_airport_delay'] is not None:
        flight_df['historical_arrival_delay'] = flight_info['historical_arrival_airport_delay']
    if 'historical_route_delay' in flight_info and flight_info['historical_route_delay'] is not None:
        flight_df['historical_route_delay'] = flight_info['historical_route_delay']
    
    # Add FAA restrictions if available
    if 'faa_restrictions' in flight_info and flight_info['faa_restrictions']:
        flight_df['has_faa_restriction'] = 1
    
    # Extract features using our extraction function
    features, _, _ = extract_features(flight_df, db)
    
    # Preprocess the features
    X_processed = preprocessor.transform(features)
    
    # Make delay probability prediction
    # Get probabilities - second column is probability of delay
    delay_probs = classifier.predict_proba(X_processed)[0]
    delay_prob = delay_probs[1] if len(delay_probs) > 1 else 0.0
    
    # Only predict delay duration if probability is significant
    estimated_delay = 0
    if delay_prob > 0.2 and regressor is not None:
        # Predict delay duration
        estimated_delay = int(max(0, regressor.predict(X_processed)[0]))
    else:
        # Simple estimation based on probability
        max_delay = 180  # Maximum delay in minutes
        estimated_delay = int(delay_prob * max_delay)
    
    # Confidence based on how far from 0.5 the probability is
    confidence = 1.0 - abs(0.5 - delay_prob) * 2 if delay_prob <= 0.5 else abs(0.5 - delay_prob) * 2
    
    # Calculate feature importances if supported by the model
    feature_importances = {}
    if hasattr(classifier, 'feature_importances_'):
        # Get feature names from the preprocessor if available
        feature_names = []
        for name, trans, cols in preprocessor.transformers_:
            if hasattr(trans, 'get_feature_names_out') and name != 'cat':  # Skip categorical for now
                feature_names.extend([f"{name}_{col}" for col in cols])
            else:
                feature_names.extend([f"{name}_{col}" for col in cols])
        
        # Match feature importances with names
        for i, imp in enumerate(classifier.feature_importances_):
            if i < len(feature_names):
                feature_importances[feature_names[i]] = float(imp)
    
    return {
        "flight_number": flight_info.get("flight_number", "Unknown"),
        "delay_probability": float(delay_prob),
        "estimated_delay_minutes": estimated_delay,
        "prediction_confidence": float(confidence),
        "feature_importances": feature_importances
    }

def initialize_model(db: Session):
    """
    Initialize the model by either loading it or training a new one.
    Returns the classifier, regressor, and preprocessor.
    """
    classifier, regressor, preprocessor = load_model()
    if classifier is None or regressor is None or preprocessor is None:
        print("Training new flight delay prediction model with enhanced features...")
        classifier, regressor, preprocessor = train_model(db)
    return classifier, regressor, preprocessor 