import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import PlayerSearch from './Search/PlayerSearch'

const Navbar = () => {
  const location = useLocation();
  
  return (
    <nav className="bg-gray-800 fixed top-0 left-0 right-0 z-50">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Left side - Navigation Links */}
          <div className="flex items-center space-x-4">
            <Link 
              to="/"
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                location.pathname === '/' 
                  ? 'bg-gray-900 text-white' 
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              }`}
            >
              Home
            </Link>
            
            <Link 
              to="/tradeanalyzer"
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                location.pathname === '/tradeanalyzer'
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              }`}
            >
              Trade Analyzer
            </Link>

            <Link 
              to="/projections"
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                location.pathname === '/projections'
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              }`}
            >
              Projections
            </Link>

            <Link 
              to="/about"
              className={`px-3 py-2 rounded-md text-sm font-medium ${
                location.pathname === '/about'
                  ? 'bg-gray-900 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              }`}
            >
              About
            </Link>
          </div>

          {/* Right side - Search */}
          <div className="flex-1 flex justify-end">
            <div className="w-full max-w-xs">
              <PlayerSearch />
            </div>
          </div>
        </div>
      </div>
    </nav>
  );
};

export default Navbar;