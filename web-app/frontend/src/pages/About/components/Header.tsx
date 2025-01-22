import React from 'react'
import { FaGithub, FaChartLine, FaCode, FaExchangeAlt } from 'react-icons/fa'

const Header = () => {
    return (
      <section id="header" className="relative min-h-[calc(100vh-4rem)] flex flex-col justify-center">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900" />
        
        <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-16">
            <h1 className="text-6xl font-extrabold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400">
              LongBall
            </h1>
            <p className="text-2xl text-gray-300 font-light max-w-3xl mx-auto">
              Open Source Baseball Career Projections
            </p>
          </div>
  
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-8 mb-16">
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-emerald-400 mb-4">
                <FaChartLine className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Career Projections</h3>
              <p className="text-gray-400">
                Complete career trajectory modeling using deep learning
              </p>
            </div>
  
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-emerald-400 mb-4">
                <FaCode className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Open Source</h3>
              <p className="text-gray-400">
                Transparent methodology and community-driven development
              </p>
            </div>
  
            <div className="bg-white/5 backdrop-blur-sm rounded-xl p-6 border border-white/10 hover:border-white/20 transition-all">
              <div className="text-emerald-400 mb-4">
                <FaExchangeAlt className="h-6 w-6" />
              </div>
              <h3 className="text-xl font-semibold text-white mb-2">Trade Analysis</h3>
              <p className="text-gray-400">
                Free tools for evaluating trades and player development
              </p>
            </div>
          </div>
  
          <div className="flex justify-center gap-6">
            <a 
              href="https://github.com/nielsjsc/LSTMLB" 
              target="_blank" 
              rel="noopener noreferrer"
              className="group px-8 py-3 rounded-md bg-emerald-500 hover:bg-emerald-400 transition-colors flex items-center gap-2"
            >
              <FaGithub className="text-xl" />
              <span>View on GitHub</span>
            </a>
            <a 
              href="#models" 
              className="px-8 py-3 rounded-md border border-white/20 hover:bg-white/5 transition-all"
            >
              Learn More
            </a>
          </div>
        </div>
      </section>
    )
}

export default Header