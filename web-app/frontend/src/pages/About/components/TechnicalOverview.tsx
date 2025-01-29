import React from 'react'
const TechnicalOverview = () => {
    return (
      <section id="technical" className="py-12">
      <div id="introduction" className="mb-12">
        <h2 className="text-2xl font-bold mb-4">How Do Baseball Predictions Work?</h2>
        <div className="prose dark:prose-invert prose-slate max-w-none">
            <p className="mb-4">
            Baseball predictions require processing vast amounts of statistical data across multiple seasons.
            While human scouts excel at qualitative analysis, our AI system's strength lies in its ability
            to process thousands of statistical relationships simultaneously, finding meaningful patterns
            in historical data to generate precise predictions.
            </p>
        </div>
        </div>
  
        <div id="learning-patterns" className="mb-12">
            <h2 className="text-2xl font-bold mb-4">Learning from History</h2>
            <div className="prose dark:prose-invert prose-slate max-w-none">
                <p className="mb-4">
                Consider Mike Trout's career progression:
                </p>
                <div className="bg-slate-100 dark:bg-slate-800 p-4 rounded-lg mb-4">
                <p className="font-mono text-sm text-slate-900 dark:text-slate-100">
                    2011: .220 AVG → 2012: .326 AVG → 2013: .323 AVG → 2014: .287 AVG
                </p>
            </div>
            <p className="mb-4">
                Our model analyzes 3-5 year sequences of player statistics, learning how early-career
                patterns translate to future performance. By processing thousands of such sequences,
                the system builds a mathematical understanding of player development trajectories.
                </p>
            </div>
            </div>
  
        <div id="memory-system" className="mb-12">
        <h2 className="text-2xl font-bold mb-4">Baseball Memory: How the AI Thinks</h2>
        <div className="prose dark:prose-invert prose-slate max-w-none">
         
            <p className="mb-4">
              Our LSTM (Long Short-Term Memory) system works like a baseball scout's brain:
            </p>
            <ul className="list-disc pl-6 mb-4 space-y-2">
              <li>
                <strong>Short-term memory:</strong> Recent performance (like last season's stats)
              </li>
              <li>
                <strong>Long-term memory:</strong> Career-long patterns (development curves)
              </li>
              <li>
                <strong>Pattern recognition:</strong> Understanding how different stats relate
                (e.g., how increased walk rate might predict power development)
              </li>
            </ul>
          </div>
        </div>
  
        <div id="attention-mechanism" className="mb-12">
          <h2 className="text-2xl font-bold mb-4">Focusing on What Matters</h2>
          <div className="prose dark:prose-invert prose-slate max-w-none">
            <p className="mb-4">
              Just as a scout might pay special attention to a player's performance against
              top pitchers or in crucial situations, our model uses an "attention mechanism"
              to focus on the most relevant parts of a player's history when making predictions.
            </p>
            <div className="bg-slate-100 dark:bg-slate-800 p-4 rounded-lg mb-4">
            <p className="mb-2 text-slate-900 dark:text-slate-100"><strong>Example: Young Power Hitter</strong></p>
            <ul className="list-disc pl-6 text-slate-900 dark:text-slate-100">
                <li>Early career: High strikeouts, low average → Less weight in predictions</li>
                <li>Recent seasons: Improved contact, maintaining power → Higher weight</li>
                <li>Key stats: Exit velocity, barrel rate trends → Special focus</li>
            </ul>
            </div>
          </div>
        </div>
  
        <div id="practical-application" className="mb-12">
          <h2 className="text-2xl font-bold mb-4">Making Predictions</h2>
          <div className="prose max-w-none dark:prose-invert">
            <p className="mb-4">
              The system combines all these elements to make predictions that account for:
            </p>
            <ul className="list-disc pl-6 mb-4">
              <li>Career trajectory and player age</li>
              <li>Recent performance changes</li>
              <li>Similar historical player patterns</li>
              <li>Complex stat interactions</li>
            </ul>
            <p className="text-sm text-gray-600 dark:text-gray-400 italic">
              This approach allows us to capture nuanced development patterns that traditional
              statistical methods might miss.
            </p>
          </div>
        </div>
      </section>
    )
  }
  
  export default TechnicalOverview