import React from 'react'
import { Link } from 'react-router-dom'
import PlayerSearch from './Search/PlayerSearch'

const Navbar = () => {
  return (
    <nav className="bg-gray-800 text-white p-4 fixed top-0 w-full z-50">
      <div className="container mx-auto flex items-center justify-between">
        <div className="flex items-center space-x-6">
          <Link to="/" className="text-xl font-bold">MLB Player Eval</Link>
          <Link to="/players" className="hover:text-gray-300">Players</Link>
          <Link to="/tradeanalyzer" className="hover:text-gray-300">Trade Analyzer</Link>
        </div>
        <div className="w-64">
          <PlayerSearch />
        </div>
      </div>
    </nav>
  )
}

export default Navbar