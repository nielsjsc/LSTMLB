import React from 'react'
import { FaGithub, FaChartLine, FaCode, FaExchangeAlt, FaStar } from 'react-icons/fa'

const Header = () => {
    return (
      <section id="header" className="flex flex-col justify-center py-20">
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h1 className="text-6xl font-extrabold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400 pb-3">
              LongBall
            </h1>
            <p className="text-2xl text-gray-300 font-light max-w-3xl mx-auto">
              Open Source Baseball Analytics Platform
            </p>
            <p className="text-lg text-gray-400 mt-4 max-w-2xl mx-auto">
              MLB projections, prospect valuations, and trade analysis - 
              free and open source for the baseball community.
            </p>
          </div>
  
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8 mb-16">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-emerald-400 mb-4">
                <FaChartLine className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">MLB Projections</h3>
              <p className="text-gray-400">
                Long-term career projections using specialized models for hitting, pitching, and fielding
              </p>
            </div>
  
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-emerald-400 mb-4">
                <FaExchangeAlt className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Trade Analysis</h3>
              <p className="text-gray-400">
                Comprehensive trade simulator with WAR projections and surplus value calculations
              </p>
            </div>

            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-emerald-400 mb-4">
                <FaStar className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Prospect Values</h3>
              <p className="text-gray-400">
                FV and ranking-based prospect valuations with dynamic graduate adjustments
              </p>
            </div>
  
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-emerald-400 mb-4">
                <FaCode className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Open Source</h3>
              <p className="text-gray-400">
                All models and code freely available for community use and contribution
              </p>
            </div>
          </div>
  
          <div className="flex justify-center">
            <a 
              href="https://github.com/nielsjsc/LSTMLB" 
              target="_blank" 
              rel="noopener noreferrer"
              className="group px-8 py-3 rounded-md bg-emerald-500 hover:bg-emerald-400 transition-colors flex items-center gap-2"
            >
              <FaGithub className="text-xl" />
              <span>View on GitHub</span>
            </a>
          </div>
        </div>
        </div>
      </section>
    )
}

export default Header