import { useState } from 'react'
import './App.css'
import Dashboard from './components/Dashboard'
import DelayPrediction from './components/DelayPrediction'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Airport Operations Dashboard</h1>
        <nav className="app-nav">
          <button 
            className={`nav-button ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            Flight Dashboard
          </button>
          <button 
            className={`nav-button ${activeTab === 'prediction' ? 'active' : ''}`}
            onClick={() => setActiveTab('prediction')}
          >
            Delay Prediction
          </button>
        </nav>
      </header>
      <main>
        {activeTab === 'dashboard' ? <Dashboard /> : <DelayPrediction />}
      </main>
      <footer className="app-footer">
        <p>&copy; {new Date().getFullYear()} Airport Operations Dashboard</p>
      </footer>
    </div>
  )
}

export default App 