import React from 'react'
import { Link } from 'react-router-dom'
import { FaGithub, FaChartLine, FaExchangeAlt, FaStar, FaInfoCircle } from 'react-icons/fa'

const Home = () => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-surface-900 via-surface-800 to-surface-900">
      <div className="max-w-7xl mx-auto px-4 py-16">
        {/* Hero Section */}
        <div className="text-center mb-20">
          <h1 className="text-6xl font-extrabold mb-6 bg-clip-text text-transparent bg-gradient-brand py-4 px-2 inline-block">
            LongBall
          </h1>
          <p className="text-2xl text-surface-300 font-light max-w-3xl mx-auto mb-8">
            Open Source Baseball Analytics Platform
          </p>
          <a 
            href="https://github.com/nielsjsc/LSTMLB" 
            target="_blank" 
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-6 py-2 rounded-md bg-white/10 hover:bg-white/20 transition-all text-white text-sm"
          >
            <FaGithub className="h-5 w-5" />
            <span>View on GitHub</span>
          </a>
        </div>

        {/* Main Navigation Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
          <Link 
            to="/projections" 
            className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-blue-500/10 to-blue-600/10 border border-blue-500/20 hover:border-blue-400/30 transition-all p-8 hover:shadow-glow"
          >
            <FaChartLine className="h-8 w-8 text-accent-blue mb-4" />
            <h2 className="text-2xl font-bold text-white mb-3">Player Projections</h2>
            <p className="text-surface-400 mb-8">
              Long-term career trajectory predictions using advanced ML models
            </p>
            <span className="text-accent-blue group-hover:text-blue-300 transition-colors flex items-center gap-2">
              View Projections →
            </span>
          </Link>

          <Link 
            to="/tradesimulator" 
            className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-brand-500/10 to-brand-600/10 border border-brand-500/20 hover:border-brand-400/30 transition-all p-8 hover:shadow-glow"
          >
            <FaExchangeAlt className="h-8 w-8 text-brand-400 mb-4" />
            <h2 className="text-2xl font-bold text-white mb-3">Trade Simulator</h2>
            <p className="text-surface-400 mb-8">
              Evaluate trades using projection-based surplus values
            </p>
            <span className="text-brand-400 group-hover:text-brand-300 transition-colors flex items-center gap-2">
              Try Simulator →
            </span>
          </Link>

          <Link 
            to="/prospects" 
            className="group relative overflow-hidden rounded-2xl bg-gradient-to-br from-purple-500/10 to-purple-600/10 border border-purple-500/20 hover:border-purple-400/30 transition-all p-8 hover:shadow-glow"
          >
            <FaStar className="h-8 w-8 text-accent-indigo mb-4" />
            <h2 className="text-2xl font-bold text-white mb-3">Prospect Rankings</h2>
            <p className="text-surface-400 mb-8">
              Consensus rankings and FV-based valuations
            </p>
            <span className="text-accent-indigo group-hover:text-purple-300 transition-colors flex items-center gap-2">
              View Prospects →
            </span>
          </Link>
        </div>

        {/* About Section */}
        <div className="flex justify-center">
          <Link 
            to="/about"
            className="group flex items-center gap-3 px-8 py-4 rounded-xl bg-white/5 hover:bg-white/10 transition-all border border-white/[0.06] hover:border-white/[0.12]"
          >
            <FaInfoCircle className="h-5 w-5 text-gray-400" />
            <span className="text-gray-300">Learn about our methodology</span>
            <svg 
              className="w-5 h-5 text-gray-400 transform group-hover:translate-x-1 transition-transform" 
              fill="none" 
              stroke="currentColor" 
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </Link>
        </div>
      </div>
    </div>
  )
}

export default Home