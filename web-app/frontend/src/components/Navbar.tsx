import React, { useState } from 'react'
import { Link, useLocation } from 'react-router-dom'
import PlayerSearch from './Search/PlayerSearch'
import { FaBars, FaTimes } from 'react-icons/fa'

const navLinks = [
  { to: '/projections', label: 'Projections' },
  { to: '/tradevalues', label: 'Trade Values' },
  { to: '/trades', label: 'Past Trades' },
  { to: '/tradesimulator', label: 'Trade Simulator' },
  { to: '/prospects', label: 'Prospects' },
  { to: '/about', label: 'About' },
];

const Navbar = () => {
  const location = useLocation();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 bg-navy-700 border-b border-navy-600">
      <div className="max-w-[90rem] mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-14">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-1.5 shrink-0 group">
            <span className="font-display text-xl font-extrabold tracking-tight text-white group-hover:opacity-80 transition-opacity">
              BASEBALL<span className="text-brand-400">VALUES</span>
            </span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-0.5 ml-6">
            {navLinks.map(({ to, label }) => {
              const isActive = location.pathname === to;
              return (
                <Link
                  key={to}
                  to={to}
                  className={`relative px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'text-white bg-white/[0.12]'
                      : 'text-navy-200 hover:text-white hover:bg-white/[0.08]'
                  }`}
                >
                  {label}
                  {isActive && (
                    <span className="absolute bottom-0 left-3 right-3 h-[2px] bg-brand-500 rounded-full" />
                  )}
                </Link>
              );
            })}
          </div>

          {/* Desktop Search */}
          <div className="hidden md:flex flex-1 justify-end ml-4">
            <div className="w-full max-w-xs">
              <PlayerSearch />
            </div>
          </div>

          {/* Mobile hamburger */}
          <button
            onClick={() => setMobileOpen(!mobileOpen)}
            className="md:hidden p-2 rounded-lg text-navy-200 hover:text-white hover:bg-white/[0.08] transition-colors"
            aria-label="Toggle navigation menu"
          >
            {mobileOpen ? <FaTimes className="h-5 w-5" /> : <FaBars className="h-5 w-5" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      {mobileOpen && (
        <div className="md:hidden border-t border-navy-600 bg-navy-700">
          <div className="px-4 py-3 space-y-1">
            {navLinks.map(({ to, label }) => {
              const isActive = location.pathname === to;
              return (
                <Link
                  key={to}
                  to={to}
                  onClick={() => setMobileOpen(false)}
                  className={`block px-3 py-2.5 rounded-lg text-sm font-medium transition-colors ${
                    isActive
                      ? 'text-white bg-white/[0.12]'
                      : 'text-navy-200 hover:text-white hover:bg-white/[0.08]'
                  }`}
                >
                  {label}
                </Link>
              );
            })}
          </div>
          <div className="px-4 pb-4">
            <PlayerSearch />
          </div>
        </div>
      )}
    </nav>
  );
};

export default Navbar;