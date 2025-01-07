import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Home from './pages/Home/Home'
import Players from './pages/Players/Players'
import TradeAnalyzer from './pages/TradeAnalyzer/TradeAnalyzer'
import PlayerDetails from './pages/PlayerDetails/PlayerDetails'

function App() {
  return (
    <Router>
      <Navbar />
      <div className="container mx-auto px-4 pt-20">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/players" element={<Players />} />
          <Route path="/tradeanalyzer" element={<TradeAnalyzer />} />
          <Route path="/player/:playerId" element={<PlayerDetails />} />
        </Routes>
      </div>
    </Router>
  )
}

export default App