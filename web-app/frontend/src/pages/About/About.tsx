import React from 'react'
import Header from './components/Header'
import TechnicalOverview from './components/TechnicalOverview'

const About = () => {
  return (
    <div className="min-h-screen w-full bg-gradient-to-b from-slate-50 to-white dark:from-slate-900 dark:to-slate-800">
      {/* Main Content */}
      <main className="w-full">
        <div className="max-w-6xl mx-auto px-8 py-20 space-y-24">
          <Header />
          <TechnicalOverview />
        </div>
      </main>
    </div>
  )
}

export default About