# Airport Operations Dashboard Frontend

This is a React-based frontend for the Airport Operations Dashboard application. It displays flight information from the backend API.

## Features

- Real-time flight information display
- Responsive design
- Automatic data refresh
- Status-based styling for flights

## Technologies Used

- React.js
- Vite
- Axios for API requests
- CSS for styling

## Setup and Installation

1. Make sure you have Node.js and npm installed
2. Install dependencies:
   ```
   npm install
   ```
3. Start the development server:
   ```
   npm run dev
   ```
4. Build for production:
   ```
   npm run build
   ```

## Backend Integration

This frontend connects to a FastAPI backend running on port 8000. Make sure the backend server is running before starting the frontend application.

The Vite configuration includes a proxy that redirects API requests to the backend server.

## Project Structure

- `src/components/Dashboard.jsx` - Main component that fetches and displays flight data
- `src/App.jsx` - Root component that includes the Dashboard
- CSS files for styling each component 