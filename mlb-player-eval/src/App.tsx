import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout/Layout'
import Players from './pages/Players/Players'
import TradeAnalyzer from './pages/TradeAnalyzer/TradeAnalyzer'
import About from './pages/About/About'

function App() {
  return (
    <Router>
      <Layout>
        <Routes>
          <Route path="/" element={
            <div>
              <h1>MLB Player Evaluation</h1>
              <p>Welcome to the MLB Player Evaluation Dashboard</p>
            </div>
          } />
          <Route path="/players" element={<Players />} />
          <Route path="/TradeAnalyzer" element={<TradeAnalyzer />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </Layout>
    </Router>
  )
}

export default App