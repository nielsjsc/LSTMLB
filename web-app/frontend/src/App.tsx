import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home/Home'
import TradeAnalyzer from './pages/TradeAnalyzer/TradeAnalyzer'
import PlayerDetails from './pages/PlayerDetails/PlayerDetails'
import Projections from './pages/Projections/Projections'
import About from './pages/About/About'

function App() {
  return (
    <Router>
      <Navbar />
      <div className="container mx-auto px-4 pt-20">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/tradeanalyzer" element={<TradeAnalyzer />} />
          <Route path="/players/:playerId" element={<PlayerDetails />} />
          <Route path="/projections" element={<Projections />} />
          <Route path="/about" element={<About />} /> 
        </Routes>
      </div>
    </Router>
  )
}

export default App