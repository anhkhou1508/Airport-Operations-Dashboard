import requests
import json
from datetime import datetime, timedelta
import time

# Wait a moment for the server to start
print("Waiting for server to be ready...")
time.sleep(5)

base_url = "http://localhost:8000"

# Current time for scheduling
now = datetime.now()
departure_time = now + timedelta(hours=4)
arrival_time = now + timedelta(hours=7)

# Create a full-featured request
features = {
    "airline": "Delta",
    "departure_airport": "ATL",
    "arrival_airport": "LAX",
    "scheduled_departure": departure_time.isoformat(),
    "scheduled_arrival": arrival_time.isoformat(),
    "departure_weather": {
        "airport_code": "ATL",
        "temperature": 32.5,
        "wind_speed": 25.3,
        "precipitation": 15.2,
        "visibility": 3.5,
        "storm_indicator": True
    },
    "arrival_weather": {
        "airport_code": "LAX",
        "temperature": 22.1,
        "wind_speed": 8.7,
        "precipitation": 0.0,
        "visibility": 10.0,
        "storm_indicator": False
    },
    "departure_congestion": {
        "departures_count": 55,
        "arrivals_count": 48,
        "congestion_level": 0.95
    },
    "arrival_congestion": {
        "departures_count": 30,
        "arrivals_count": 25,
        "congestion_level": 0.7
    },
    "is_holiday": True,
    "aircraft_info": {
        "aircraft_type": "Boeing 737-800",
        "aircraft_age": 12.5
    },
    "previous_flight_delay": 45,
    "historical_airline_delay": 22.5,
    "historical_departure_airport_delay": 18.7,
    "historical_arrival_airport_delay": 15.3,
    "historical_route_delay": 25.2
}

print("Sending request with the following features:")
print(json.dumps(features, indent=2))

try:
    # First test health endpoint
    health_response = requests.get(f"{base_url}/health")
    print(f"\nHealth check: {health_response.status_code}")
    if health_response.status_code == 200:
        print(health_response.json())
    else:
        print("Health check failed. Server may not be running properly.")
        exit(1)
    
    # Now test the prediction endpoint
    response = requests.post(
        f"{base_url}/predict-delay", 
        json=features,
        timeout=10  # 10 second timeout
    )
    
    print(f"\nPrediction response status code: {response.status_code}")
    
    if response.status_code == 200:
        result = response.json()
        print("\nPrediction Result:")
        print(json.dumps(result, indent=2))
        
        if 'feature_importances' in result and result['feature_importances']:
            print("\nFeature Importances Summary:")
            sorted_features = sorted(
                result['feature_importances'].items(),
                key=lambda x: x[1],
                reverse=True
            )
            for feature, importance in sorted_features[:10]:  # Top 10 features
                print(f"  {feature}: {importance:.4f}")
                
            # Check which feature types are most important
            weather_features = [f for f in sorted_features if 'weather' in f[0]]
            congestion_features = [f for f in sorted_features if 'congestion' in f[0]]
            historical_features = [f for f in sorted_features if 'historical' in f[0]]
            aircraft_features = [f for f in sorted_features if 'aircraft' in f[0]]
            
            print("\nFeature Type Analysis:")
            print(f"  Weather features: {len(weather_features)} in top features")
            print(f"  Congestion features: {len(congestion_features)} in top features")
            print(f"  Historical features: {len(historical_features)} in top features")
            print(f"  Aircraft features: {len(aircraft_features)} in top features")
            
            # Conclusion
            print("\nCONCLUSION:")
            if len(weather_features) + len(congestion_features) + len(historical_features) + len(aircraft_features) > 0:
                print("✅ The model is using the new features in its predictions.")
                highest_feature = sorted_features[0]
                print(f"The most important feature is: {highest_feature[0]} with importance {highest_feature[1]:.4f}")
            else:
                print("❌ The new features don't appear to be significantly influencing the prediction.")
        else:
            print("\n❌ No feature importances returned. The model may not be properly configured to use the new features.")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"Error making request: {e}") 