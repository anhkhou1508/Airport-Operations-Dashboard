import { useState } from 'react';
import axios from 'axios';
import './Dashboard.css';

const DelayPrediction = () => {
  const [formData, setFormData] = useState({
    airline: '',
    departure_airport: '',
    arrival_airport: '',
    scheduled_departure: '',
    scheduled_arrival: ''
  });
  
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData({
      ...formData,
      [name]: value
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Format datetime strings for API
    const submissionData = {
      ...formData,
      scheduled_departure: new Date(formData.scheduled_departure).toISOString(),
      scheduled_arrival: new Date(formData.scheduled_arrival).toISOString()
    };

    try {
      setLoading(true);
      setError(null);
      const response = await axios.post('/predict-delay', submissionData);
      setPrediction(response.data);
    } catch (err) {
      console.error('Error predicting delay:', err);
      setError('Failed to get delay prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  // Format probability as percentage
  const formatProbability = (probability) => {
    return (probability * 100).toFixed(1) + '%';
  };

  return (
    <div className="prediction-container">
      <h2>Flight Delay Prediction</h2>
      
      <form onSubmit={handleSubmit} className="prediction-form">
        <div className="form-group">
          <label htmlFor="airline">Airline:</label>
          <input
            type="text"
            id="airline"
            name="airline"
            value={formData.airline}
            onChange={handleChange}
            required
            placeholder="e.g. American"
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="departure_airport">Departure Airport (Code):</label>
          <input
            type="text"
            id="departure_airport"
            name="departure_airport"
            value={formData.departure_airport}
            onChange={handleChange}
            required
            placeholder="e.g. JFK"
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="arrival_airport">Arrival Airport (Code):</label>
          <input
            type="text"
            id="arrival_airport"
            name="arrival_airport"
            value={formData.arrival_airport}
            onChange={handleChange}
            required
            placeholder="e.g. LAX"
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="scheduled_departure">Scheduled Departure:</label>
          <input
            type="datetime-local"
            id="scheduled_departure"
            name="scheduled_departure"
            value={formData.scheduled_departure}
            onChange={handleChange}
            required
          />
        </div>
        
        <div className="form-group">
          <label htmlFor="scheduled_arrival">Scheduled Arrival:</label>
          <input
            type="datetime-local"
            id="scheduled_arrival"
            name="scheduled_arrival"
            value={formData.scheduled_arrival}
            onChange={handleChange}
            required
          />
        </div>
        
        <button type="submit" className="submit-button" disabled={loading}>
          {loading ? 'Predicting...' : 'Predict Delay'}
        </button>
      </form>

      {error && (
        <div className="error-message">
          <p>{error}</p>
        </div>
      )}

      {prediction && !error && (
        <div className="prediction-results">
          <h3>Prediction Results:</h3>
          <div className="result-card">
            <div className="result-item">
              <span className="label">Flight:</span>
              <span className="value">{prediction.flight_number}</span>
            </div>
            <div className="result-item">
              <span className="label">Delay Probability:</span>
              <span className="value" style={{ color: prediction.delay_probability > 0.5 ? 'red' : 'green' }}>
                {formatProbability(prediction.delay_probability)}
              </span>
            </div>
            <div className="result-item">
              <span className="label">Estimated Delay:</span>
              <span className="value">
                {prediction.estimated_delay_minutes} minutes
              </span>
            </div>
            <div className="result-item">
              <span className="label">Confidence:</span>
              <span className="value">
                {formatProbability(prediction.prediction_confidence)}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DelayPrediction; 