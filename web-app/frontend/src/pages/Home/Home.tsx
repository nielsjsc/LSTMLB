import React from 'react'
import { Link } from 'react-router-dom'

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-blue-50">
      <div className="max-w-6xl mx-auto px-4 py-16">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <div className="mb-6">
            <h1 className="text-6xl font-extrabold mb-2">
              <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-600 via-indigo-600 to-blue-600">
                LSTMLB
              </span>
            </h1>
            <p className="text-lg text-gray-500">Powered by Machine Learning</p>
          </div>
          <p className="text-xl text-gray-600 max-w-2xl mx-auto">
            Advanced neural networks meet America's pastime. Make better decisions with AI-driven player projections and trade analysis.
          </p>
        </div>

        {/* Feature Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <Link to="/tradeanalyzer" 
            className="group relative overflow-hidden bg-white p-8 rounded-2xl shadow-md hover:shadow-xl transition-all duration-300 border border-gray-100">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-50 to-indigo-50 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative">
              <h2 className="text-2xl font-bold mb-4 text-gray-800 group-hover:text-blue-600 transition-colors">
                Trade Simulator
              </h2>
              <p className="text-gray-600 mb-6">
                Leverage our LSTM-powered models to evaluate trades with precision. Get instant insights into player values and team dynamics.
              </p>
              <div className="flex items-center text-blue-600 font-medium group-hover:text-blue-700">
                Analyze Trades
                <svg className="w-5 h-5 ml-2 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </div>
            </div>
          </Link>

          <Link to="/projections" 
            className="group relative overflow-hidden bg-white p-8 rounded-2xl shadow-md hover:shadow-xl transition-all duration-300 border border-gray-100">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-50 to-indigo-50 opacity-0 group-hover:opacity-100 transition-opacity"></div>
            <div className="relative">
              <h2 className="text-2xl font-bold mb-4 text-gray-800 group-hover:text-blue-600 transition-colors">
                Player Projections
              </h2>
              <p className="text-gray-600 mb-6">
                Deep learning models trained on decades of MLB data provide cutting-edge player projections and performance insights.
              </p>
              <div className="flex items-center text-blue-600 font-medium group-hover:text-blue-700">
                View Projections
                <svg className="w-5 h-5 ml-2 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
                </svg>
              </div>
            </div>
          </Link>
        </div>
      </div>
    </div>
  )
}

export default Home