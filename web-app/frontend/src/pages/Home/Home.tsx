import React from 'react'
import { Link } from 'react-router-dom'
import { FaGithub, FaChartLine, FaCode, FaExchangeAlt } from 'react-icons/fa'

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto px-4 py-16">
        {/* Hero Section */}
        <div className="text-center mb-16">
          <h1 className="text-6xl font-extrabold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400 py-4 px-2 inline-block">
            LongBall
          </h1>
          <p className="text-2xl text-gray-300 font-light max-w-3xl mx-auto mb-12">
            Open Source Baseball Projections
          </p>
          
          {/* Feature Cards */}
          <div className="grid md:grid-cols-3 gap-6 mb-16">
            <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-emerald-400 mb-4">
                <FaChartLine className="h-6 w-6 mx-auto" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Projections</h3>
              <p className="text-gray-400 text-sm">
                Complete career trajectory modeling using deep learning
              </p>
            </div>

            <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-emerald-400 mb-4">
                <FaCode className="h-6 w-6 mx-auto" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Open Source</h3>
              <p className="text-gray-400 text-sm">
                Transparent methodology and community-driven development
              </p>
            </div>

            <div className="bg-white/5 backdrop-blur-sm rounded-lg p-6 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-emerald-400 mb-4">
                <FaExchangeAlt className="h-6 w-6 mx-auto" />
              </div>
              <h3 className="text-lg font-semibold text-white mb-2">Trade Analysis</h3>
              <p className="text-gray-400 text-sm">
                Free tools for evaluating trades and player development
              </p>
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex justify-center gap-4 mb-16">
            <Link 
              to="/tradeanalyzer"
              className="px-6 py-2 rounded-md bg-emerald-500 hover:bg-emerald-400 transition-colors text-sm"
            >
              Try Trade Analyzer
            </Link>
            <a 
              href="https://github.com/nielsjsc/LSTMLB" 
              target="_blank" 
              rel="noopener noreferrer"
              className="px-6 py-2 rounded-md border border-white/20 hover:bg-white/5 transition-all text-white text-sm flex items-center gap-2"
            >
              <FaGithub className="h-5 w-5" />
              <span>View on GitHub</span>
            </a>
          </div>
        </div>

        {/* Navigation Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <Link to="/projections" className="group bg-white/5 backdrop-blur-sm p-8 rounded-xl border border-white/10 hover:border-white/20 transition-all">
            <h2 className="text-2xl font-bold mb-4 text-white">Player Projections</h2>
            <p className="text-gray-400 mb-6">
              Explore career projections powered by deep learning models trained on decades of MLB data.
            </p>
            <div className="flex items-center text-emerald-400 font-medium">
              View Projections
              <svg className="w-5 h-5 ml-2 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </div>
          </Link>

          <Link to="/about" className="group bg-white/5 backdrop-blur-sm p-8 rounded-xl border border-white/10 hover:border-white/20 transition-all">
            <h2 className="text-2xl font-bold mb-4 text-white">About LongBall</h2>
            <p className="text-gray-400 mb-6">
              Learn about our methodology, including LSTM neural networks and statistical analysis.
            </p>
            <div className="flex items-center text-emerald-400 font-medium">
              Learn More
              <svg className="w-5 h-5 ml-2 transform group-hover:translate-x-1 transition-transform" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </div>
          </Link>
        </div>
      </div>
    </div>
  )
}

export default Home