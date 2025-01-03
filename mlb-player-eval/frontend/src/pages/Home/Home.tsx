import React from 'react'
import { Link } from 'react-router-dom'

const Home = () => {
  return (
    <div className="max-w-4xl mx-auto py-12">
      <h1 className="text-4xl font-bold mb-8">MLB Player Evaluation Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Link to="/tradeanalyzer" 
          className="p-6 border rounded-lg hover:shadow-lg transition">
          <h2 className="text-2xl font-bold mb-4">Trade Analyzer</h2>
          <p>Analyze and evaluate potential MLB trades</p>
        </Link>
        <Link to="/players" 
          className="p-6 border rounded-lg hover:shadow-lg transition">
          <h2 className="text-2xl font-bold mb-4">Player Database</h2>
          <p>Browse and search MLB player projections</p>
        </Link>
      </div>
    </div>
  )
}

export default Home