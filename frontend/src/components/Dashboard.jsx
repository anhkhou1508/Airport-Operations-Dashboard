import { useState, useEffect } from 'react';
import axios from 'axios';
import './Dashboard.css';

const Dashboard = () => {
  const [flights, setFlights] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchFlights = async () => {
      try {
        setLoading(true);
        const response = await axios.get('/flights');
        setFlights(response.data);
        setError(null);
      } catch (err) {
        console.error('Error fetching flights:', err);
        setError('Failed to fetch flight data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchFlights();
    
    // Refresh data every 1 minute
    const intervalId = setInterval(fetchFlights, 60000);
    
    return () => clearInterval(intervalId);
  }, []);

  // Format date to a readable string
  const formatDateTime = (dateTimeStr) => {
    const date = new Date(dateTimeStr);
    return date.toLocaleString();
  };

  if (loading) {
    return (
      <div className="loading-container">
        <div className="loading-spinner"></div>
        <p>Loading flight data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-container">
        <h2>Error</h2>
        <p>{error}</p>
        <button onClick={() => window.location.reload()}>Try Again</button>
      </div>
    );
  }

  return (
    <div className="dashboard-container">
      <h2>Flight Schedule</h2>
      
      {flights.length === 0 ? (
        <p className="no-flights">No flights scheduled at this time.</p>
      ) : (
        <div className="table-container">
          <table className="flights-table">
            <thead>
              <tr>
                <th>Flight Number</th>
                <th>Departure Time</th>
                <th>Status</th>
                <th>Airline</th>
                <th>Destination</th>
                <th>Gate</th>
              </tr>
            </thead>
            <tbody>
              {flights.map((flight) => (
                <tr key={flight.id} className={`status-${flight.status.toLowerCase().replace(' ', '-')}`}>
                  <td>{flight.flight_number}</td>
                  <td>{formatDateTime(flight.scheduled_departure)}</td>
                  <td className="status-cell">{flight.status}</td>
                  <td>{flight.airline}</td>
                  <td>{flight.arrival_airport}</td>
                  <td>{flight.gate || 'TBA'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default Dashboard; 