import React, { useState } from 'react'
import { FaGithub, FaChevronDown } from 'react-icons/fa'

const CollapsibleSection = ({ title, defaultOpen = false, children }: { title: string; defaultOpen?: boolean; children: React.ReactNode }) => (
  <details className="mb-6 group border border-gray-200 rounded-lg bg-white overflow-hidden shadow-sm" open={defaultOpen}>
    <summary className="cursor-pointer px-6 py-4 font-bold text-gray-900 font-display flex justify-between items-center bg-gray-50 hover:bg-gray-100 transition-colors list-none [&::-webkit-details-marker]:hidden">
      <span className="text-lg">{title}</span>
      <FaChevronDown className="h-4 w-4 text-gray-500 group-open:rotate-180 transition-transform duration-200" />
    </summary>
    <div className="p-6 space-y-4 text-[15px] text-gray-600 leading-relaxed border-t border-gray-200">
      {children}
    </div>
  </details>
)

type FeedbackStatus = 'idle' | 'submitting' | 'success' | 'error'

const FeedbackForm = () => {
  const [name, setName] = useState('')
  const [email, setEmail] = useState('')
  const [category, setCategory] = useState('general')
  const [message, setMessage] = useState('')
  const [status, setStatus] = useState<FeedbackStatus>('idle')
  const [errorMsg, setErrorMsg] = useState('')

  const handleSubmit = async () => {
    if (!message.trim()) return

    setStatus('submitting')
    setErrorMsg('')

    try {
      const res = await fetch('https://formspree.io/f/mbdezljq', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ name, email, category, message }),
      })

      if (res.ok) {
        setStatus('success')
        setName('')
        setEmail('')
        setCategory('general')
        setMessage('')
      } else {
        const data = await res.json()
        setErrorMsg(data?.errors?.[0]?.message ?? 'Something went wrong. Please try again.')
        setStatus('error')
      }
    } catch {
      setErrorMsg('Network error. Please check your connection and try again.')
      setStatus('error')
    }
  }

  if (status === 'success') {
    return (
      <div className="rounded-lg border border-green-200 bg-green-50 p-6 text-center">
        <p className="text-green-800 font-medium text-sm">Thanks for the feedback. It is genuinely appreciated.</p>
        <button
          onClick={() => setStatus('idle')}
          className="mt-3 text-xs text-green-700 underline underline-offset-2 hover:text-green-900"
        >
          Submit another
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">Name (optional)</label>
          <input
            type="text"
            value={name}
            onChange={e => setName(e.target.value)}
            placeholder="Your name"
            className="w-full rounded-md border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-gray-400 focus:outline-none focus:ring-0"
          />
        </div>
        <div>
          <label className="block text-xs font-medium text-gray-500 mb-1">Email (optional)</label>
          <input
            type="email"
            value={email}
            onChange={e => setEmail(e.target.value)}
            placeholder="you@example.com"
            className="w-full rounded-md border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-gray-400 focus:outline-none focus:ring-0"
          />
        </div>
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-500 mb-1">Category</label>
        <select
          value={category}
          onChange={e => setCategory(e.target.value)}
          className="w-full rounded-md border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 focus:border-gray-400 focus:outline-none focus:ring-0"
        >
          <option value="general">General feedback</option>
          <option value="projections">Projection accuracy</option>
          <option value="trade-values">Trade values</option>
          <option value="bug">Bug report</option>
          <option value="feature">Feature request</option>
          <option value="models">Model / methodology</option>
        </select>
      </div>

      <div>
        <label className="block text-xs font-medium text-gray-500 mb-1">
          Message <span className="text-gray-400">(required)</span>
        </label>
        <textarea
          value={message}
          onChange={e => setMessage(e.target.value)}
          rows={5}
          placeholder="What is on your mind? Specific players you think are misvalued, methodology questions, or bug reports. Anything goes."
          className="w-full rounded-md border border-gray-200 bg-white px-3 py-2 text-sm text-gray-900 placeholder-gray-400 focus:border-gray-400 focus:outline-none focus:ring-0 resize-none"
        />
      </div>

      {status === 'error' && (
        <p className="text-xs text-red-600">{errorMsg}</p>
      )}

      <button
        onClick={handleSubmit}
        disabled={status === 'submitting' || !message.trim()}
        className="px-4 py-2 rounded-md bg-gray-900 hover:bg-gray-800 disabled:bg-gray-300 disabled:cursor-not-allowed transition-colors text-white text-sm font-medium"
      >
        {status === 'submitting' ? 'Sending…' : 'Send feedback'}
      </button>
    </div>
  )
}

const About = () => {
  return (
    <div className="min-h-screen w-full bg-[#F5F3EE]">
      <main className="w-full">
        <div className="max-w-3xl mx-auto px-6 py-12">

          {/* Header */}
          <div className="mb-10 text-center">
            <h1 className="font-display text-4xl font-extrabold text-gray-900 tracking-tight mb-3">About LongBall</h1>
            <p className="text-lg text-gray-500 mb-6">
              An open-source MLB player projection system forecasting performance 15 years into the future, featuring quantified trade value, contract analysis, and surplus value modeling.
            </p>
            <a
              href="https://github.com/nielsjsc/LSTMLB"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 px-4 py-2 rounded-md bg-gray-900 hover:bg-gray-800 transition-colors text-white text-sm font-medium"
            >
              <FaGithub className="h-4 w-4" />
              <span>View on GitHub</span>
            </a>
          </div>

          <CollapsibleSection title="Share Feedback" defaultOpen>
            <p className="text-[15px] text-gray-500 mb-4">
              Found something off? Have a methodology question or feature idea? Your feedback helps improve the system.
            </p>
            <FeedbackForm />
          </CollapsibleSection>

          <CollapsibleSection title="Data Acquisition Pipeline">
            <p>LongBall ingests and normalizes data from authoritative sources to power all downstream projections.</p>
            <ul className="list-disc pl-5 space-y-2 mt-2">
              <li><strong>FanGraphs:</strong> Supplies core batting, pitching, and fielding rate statistics.</li>
              <li><strong>Statcast (Baseball Savant):</strong> Provides batted-ball and pitch-level metrics from 2016 onward.</li>
              <li><strong>MLB Stats API:</strong> Feeds rosters, game activity, and positional information.</li>
              <li><strong>Spotrac & Cot's Baseball:</strong> Delivers current contract details and historical salary data.</li>
              <li><strong>Prospect Data:</strong> Combines FanGraphs Future Value (FV) grades and MLB.com consensus rankings.</li>
              <li><strong>Identity Resolution:</strong> Unifies player identities across all platforms using crosswalk tables and fuzzy-matching.</li>
            </ul>
          </CollapsibleSection>

          <CollapsibleSection title="The Marcel Projection System">
            <p>We utilize the Marcel methodology to avoid overfitting on small samples. The process follows five strict steps:</p>
            <ul className="list-disc pl-5 space-y-2 mt-2">
              <li><strong>Step 1: Weighted Baseline.</strong> Weights a player's recent seasons at a 5/4/3 ratio.</li>
              <li><strong>Step 2: Reliability Shrinkage.</strong> Uses James-Stein Bayesian regression to pull observed stats toward the league average based on sample size.</li>
              <li><strong>Step 3: Multivariate Adjustment.</strong> Blends the baseline with a static equation derived from underlying physical Statcast metrics (like velocity and Stuff+).</li>
              <li><strong>Step 4: Component Validation.</strong> Mathematically reconstructs counting stats from rates to guarantee internal consistency.</li>
              <li><strong>Step 5: Empirical Aging Curves.</strong> Applies historically fitted aging curves to model physical decline while accounting for survivorship bias.</li>
            </ul>
            <p className="mt-3"><strong>Note:</strong> All statistics are neutralized for park factors prior to projection and adjusted for upcoming team ballparks afterward.</p>
          </CollapsibleSection>

          <CollapsibleSection title="Wins Above Replacement (WAR)">
            <p>WAR connects our projections to financial value using the FanGraphs formula with role-specific adjustments.</p>
            <ul className="list-disc pl-5 space-y-2 mt-2">
              <li><strong>Batter WAR:</strong> Sums Weighted Runs Above Average (wRAA), Baserunning (BsR), Fielding, and Positional Adjustments. Projected at a flat 150 games (135 for catchers).</li>
              <li><strong>Pitcher WAR:</strong> Calculated using FIP and a Dynamic Runs Per Win (RPW) metric. Elite pitchers actively suppress the run environment, increasing the value of the runs they save.</li>
              <li><strong>Defense & Baserunning:</strong> Relies on park-adjusted Statcast fielding run values and baserunning metrics.</li>
            </ul>
          </CollapsibleSection>

          <CollapsibleSection title="Trade Value Determination">
            <p>Trade value quantifies a player's worth on the market by calculating their surplus value (Production Value minus Contract Cost).</p>
            <ul className="list-disc pl-5 space-y-2 mt-2">
              <li><strong>WAR-to-Dollar Conversion:</strong> Utilizes a convex power-law model. Elite players command a superlinear premium. Values are inflated by 4% annually.</li>
              <li><strong>Contract Costs:</strong> Pre-arbitration is set to $720K. Arbitration is estimated at 15%, 25%, and 40% of production value. Free agents use actual Spotrac contract data.</li>
              <li><strong>Option Evaluation:</strong> The system automatically exercises player opt-outs if mathematical surplus dictates it, and declines team options yielding negative surplus.</li>
              <li><strong>Surplus Aggregation:</strong> Sums total value over the contract period without Net Present Value (NPV) discounting.</li>
            </ul>
          </CollapsibleSection>

          <CollapsibleSection title="Prospect Valuations & MiLB Translation">
            <p>Prospects rely on scouting grades and translated minor league statistics before transitioning to standard projections.</p>
            <ul className="list-disc pl-5 space-y-2 mt-2">
              <li><strong>FV-Based Valuation:</strong> FanGraphs Future Value grades (20 to 80 scale) are converted to base dollar amounts, enhanced by a Top 100 ranking multiplier.</li>
              <li><strong>MiLB Translation:</strong> AAA translates at roughly 0.7x to MLB, and Single-A at 0.35x.</li>
              <li><strong>MLB Transition:</strong> Valuations smoothly transition from FV-based to projection-based over a player's first 300 games (batters) or 45 starts (pitchers).</li>
            </ul>
          </CollapsibleSection>

          <CollapsibleSection title="Daily Pipeline & Rest-of-Season Blending">
            <p>An automated pipeline orchestrates the workflow every day at 6 AM.</p>
            <ul className="list-disc pl-5 space-y-2 mt-2">
              <li><strong>Automated Ingestion:</strong> Scrapes rosters, live stats, and new financial transactions.</li>
              <li><strong>ROS Blending:</strong> During the active season, preseason Marcel projections act as a Bayesian prior. Live in-season stats are regressed against this prior based on accumulated playing time.</li>
              <li><strong>Deployment:</strong> Validates outputs and seamlessly swaps the database into the production environment.</li>
            </ul>
          </CollapsibleSection>

          <CollapsibleSection title="Limitations & Caveats">
            <p>Our model is built on strict statistical expectations, which comes with inherent limitations.</p>
            <ul className="list-disc pl-5 space-y-2 mt-2">
              <li><strong>Uncertainty:</strong> Forecasting 15 years ahead is highly volatile. The numbers are baselines for valuation, not guaranteed predictions.</li>
              <li><strong>Playing Time:</strong> Health, injuries, and role changes are not explicitly modeled.</li>
              <li><strong>Mechanics:</strong> Mid-career mechanical adjustments or pitch mix shifts cannot be predicted.</li>
              <li><strong>Market Nuance:</strong> Individual front offices may value players differently than our historical average model suggests.</li>
              <li><strong>NPV:</strong> We do not discount future dollars. This may slightly overvalue long-term assets compared to teams with strict short-term cash flow constraints.</li>
            </ul>
          </CollapsibleSection>

          <CollapsibleSection title="Technical Stack">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">Data & Backend</h3>
                <p className="text-sm">Python 3.14, pandas, pybaseball, FastAPI, SQLAlchemy 2.0, PostgreSQL, Alembic</p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">Frontend</h3>
                <p className="text-sm">React 18, TypeScript 5.3, Vite, TailwindCSS, TanStack Query, React Router</p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">Machine Learning</h3>
                <p className="text-sm">PyTorch, NumPyro, scikit-learn</p>
              </div>
              <div>
                <h3 className="font-semibold text-gray-900 mb-1">Infrastructure</h3>
                <p className="text-sm">Netlify, Render, GitHub Actions, Docker</p>
              </div>
            </div>
          </CollapsibleSection>

          {/* Footer */}
          <div className="text-sm text-gray-400 pt-6 mt-8 border-t border-gray-200 text-center">
            All models, projections, and data pipelines are open source under the MIT license. Contributions and feedback are welcome on GitHub. For detailed technical documentation, visit the <a href="https://github.com/nielsjsc/LSTMLB/tree/main/docs" target="_blank" rel="noopener noreferrer" className="text-blue-600 hover:text-blue-800 transition-colors">docs folder</a> in the repository.
          </div>

        </div>
      </main>
    </div>
  )
}

export default About