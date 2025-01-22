import React, { useEffect } from 'react'
import { useLocation } from 'react-router-dom'
import Header from './components/Header'
import TechnicalOverview from './components/TechnicalOverview'
import ModelDetails from './components/ModelDetails'
import DataAnalysis from './components/DataAnalysis'

const About = () => {
  const { hash } = useLocation();

  useEffect(() => {
    if (hash) {
      const element = document.querySelector(hash);
      if (element) {
        const yOffset = -100;
        const y = element.getBoundingClientRect().top + window.pageYOffset + yOffset;
        window.scrollTo({ top: y, behavior: 'smooth' });
      }
    }
  }, [hash]);

  return (
    <div className="max-w-7xl mx-auto pt-16">
      <div className="flex">
        {/* Sidebar Navigation */}
        <div className="w-64 fixed h-screen overflow-y-auto pt-4 px-4 border-r bg-gray-50 dark:bg-gray-800">
          <nav className="space-y-1">
            <div className="pb-2">
              <a href="/about#header" className="block py-2 text-gray-900 dark:text-gray-100 hover:text-blue-600">
                Overview
              </a>
            </div>
            
            <div className="pb-2">
              <div className="text-gray-900 dark:text-gray-100 py-2">Technical Details</div>
              <div className="pl-4 space-y-1">
                <a href="/about#lstm" className="block py-1 text-gray-600 dark:text-gray-300 hover:text-blue-600">
                  LSTM Networks
                </a>
                <a href="/about#deeplearning" className="block py-1 text-gray-600 dark:text-gray-300 hover:text-blue-600">
                  Deep Learning
                </a>
                <a href="/about#architecture" className="block py-1 text-gray-600 dark:text-gray-300 hover:text-blue-600">
                  Architecture
                </a>
              </div>
            </div>

            <div className="pb-2">
              <div className="text-gray-900 dark:text-gray-100 py-2">Models</div>
              <div className="pl-4 space-y-1">
                <a href="/about#position-models" className="block py-1 text-gray-600 dark:text-gray-300 hover:text-blue-600">
                  Position Models
                </a>
                <a href="/about#training" className="block py-1 text-gray-600 dark:text-gray-300 hover:text-blue-600">
                  Training Process
                </a>
                <a href="/about#validation" className="block py-1 text-gray-600 dark:text-gray-300 hover:text-blue-600">
                  Validation
                </a>
              </div>
            </div>

            <div className="pb-2">
              <div className="text-gray-900 dark:text-gray-100 py-2">Analysis</div>
              <div className="pl-4 space-y-1">
                <a href="/about#results" className="block py-1 text-gray-600 dark:text-gray-300 hover:text-blue-600">
                  Results
                </a>
                <a href="/about#comparisons" className="block py-1 text-gray-600 dark:text-gray-300 hover:text-blue-600">
                  Comparisons
                </a>
              </div>
            </div>
          </nav>
        </div>

        {/* Main Content */}
        <div className="ml-64 flex-1 p-8">
          <main className="space-y-16">
            <Header />
            <TechnicalOverview />
            <ModelDetails />
            <DataAnalysis />
          </main>
        </div>
      </div>
    </div>
  )
}

export default About