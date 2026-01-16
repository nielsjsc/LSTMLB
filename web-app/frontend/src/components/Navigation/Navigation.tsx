import React from 'react'
import { Link } from 'react-router-dom'

const Navigation = () => {
  return (
    <nav className="bg-gray-800 p-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between">
          <Link to="/" className="text-white font-bold">
            MLB Player Evaluation
          </Link>
          <div className="space-x-4">
            <Link to="/players" className="text-white">Players</Link>
            <Link to="/TradeSimulator" className="text-white">Trade Simulator</Link>
            <Link to="/projections" className="text-white">Projections</Link>
            <Link to="/prospects" className="text-white">Prospects</Link>
            <Link to="/about" className="text-white">About</Link>
            <Link to="/tradevalues" className="text-white">Trade Values</Link>
          </div>
        </div>
      </div>
    </nav>
  )
}

export default Navigation