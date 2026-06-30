import React, { useState, useEffect } from 'react'
import { FaChevronDown } from 'react-icons/fa'

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
  useEffect(() => {
    document.title = 'About | BaseballValues'
  }, [])

  return (
    <div className="min-h-screen w-full bg-[#F5F3EE]">
      <main className="w-full">
        <div className="max-w-3xl mx-auto px-6 py-12">

          {/* Header */}
          <div className="mb-10 text-center">
            <h1 className="font-display text-4xl font-extrabold text-gray-900 tracking-tight mb-3">
              About BaseballValues
            </h1>
            <p className="text-lg text-gray-500 mb-6">
              An MLB player projection system forecasting performance up to 15 years into the future, featuring quantified trade value, contract analysis, and surplus value modeling.
            </p>
          </div>

          <CollapsibleSection title="Share Feedback" defaultOpen>
            <p className="text-[15px] text-gray-500 mb-4">
              Found something off? Have a methodology question or feature idea? Your feedback helps improve the system.
            </p>
            <FeedbackForm />
          </CollapsibleSection>

          <CollapsibleSection title="Data Acquisition Pipeline">
            <p>BaseballValues ingests and normalizes data from authoritative sources to power all downstream projections.</p>
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
            <p>Projections start with the Marcel methodology, a deliberately simple approach built to avoid overfitting on small samples. A player's three most recent seasons are weighted 5/4/3, so the most recent year counts the most.</p>
            <p>From there, we apply Bayesian shrinkage (the James-Stein method) to pull each player's stats toward the league average. The less playing time someone has, the harder their numbers get pulled toward that average, which keeps small samples from producing wild, unreliable projections.</p>
            <p>That baseline is then blended with a separate equation built from underlying Statcast metrics, things like exit velocity and Stuff+, since those tend to predict future performance better than results-based stats alone. We also rebuild counting stats like hits and strikeouts from the projected rates, so everything stays internally consistent, and apply aging curves fitted to historical player data to account for typical decline as players get older.</p>
            <p>All of this happens with park effects stripped out first, then reapplied based on each player's actual home park going forward.</p>
          </CollapsibleSection>

          <CollapsibleSection title="Wins Above Replacement (WAR)">
            <p>WAR is what connects a projection to an actual dollar value, and we calculate it using FanGraphs' formula with a few role-specific adjustments.</p>
            <p>For batters, that means combining weighted runs above average, baserunning, fielding, and a positional adjustment, projected over a flat 150 games (135 for catchers, since they typically play less). For pitchers, we use FIP along with a dynamic runs-per-win conversion: elite pitchers tend to suppress scoring more than an average pitcher would, which makes each run they save slightly more valuable. Fielding and baserunning numbers are both park-adjusted using Statcast data.</p>
          </CollapsibleSection>

          <CollapsibleSection title="Trade Value Determination">
            <p>Trade value is essentially surplus value: what a player is projected to produce minus what they actually cost. We convert WAR into dollars using a power-law curve rather than a flat rate, since elite players tend to be worth disproportionately more than their raw WAR total would suggest under a simple linear model. That dollar-per-WAR rate also gets inflated 4% each year to keep pace with the market.</p>
            <p>On the cost side, pre-arbitration players are set at a flat $720K, arbitration-eligible players are estimated at 15%, 25%, or 40% of their production value depending on service time, and free agents use real contract figures pulled from Spotrac. Player opt-outs and team options are handled automatically too: the model exercises an opt-out when the math favors the player, and assumes a team declines any option that would create negative surplus. Total surplus is then summed across the life of the contract, without discounting future dollars to present value.</p>
          </CollapsibleSection>

          <CollapsibleSection title="Prospect Valuations & MiLB Translation">
            <p>Before a prospect has enough MLB performance to project directly, we value them using FanGraphs' Future Value (FV) grades, the 20-80 scouting scale, converted into a base dollar figure and boosted slightly if they rank in a given outlet's Top 100. Minor league stats are also translated to an MLB-equivalent level to make them comparable: roughly 0.7x at AAA and 0.35x at Single-A, reflecting the jump in competition at each level.</p>
            <p>Once a prospect debuts, their valuation doesn't flip over all at once. It shifts gradually from the FV-based estimate to a standard performance-based projection over their first 300 games as a batter, or 45 starts as a pitcher.</p>
          </CollapsibleSection>

          <CollapsibleSection title="Daily Pipeline & Rest-of-Season Blending">
            <p>Everything runs on an automated pipeline each morning at 6 AM. It pulls in updated rosters, the previous day's stats, and any new contract or transaction data.</p>
            <p>During the season, preseason Marcel projections act as a starting point, and as players accumulate more games, their live performance gets weighted more heavily against that starting point. That means projections gradually shift to reflect what's actually happening on the field rather than jumping around after every game. Once the day's numbers are validated, they're swapped into production automatically.</p>
          </CollapsibleSection>

          <CollapsibleSection title="Limitations & Caveats">
            <p>Like any statistical model, this one has real limits worth knowing about. Forecasting a player's value 15 years out is inherently volatile, so these numbers should be treated as a baseline for comparison, not a guarantee of what will actually happen.</p>
            <p>Injuries, health, and role changes aren't explicitly modeled, and the system has no way to anticipate a mid-career swing change or a pitcher reworking their pitch mix. Individual front offices also weigh players differently than a historical-average model will, since philosophy varies team to team. And because we don't discount future dollars to present value, long-term contracts may come out looking slightly more valuable here than they would to a team with tighter short-term cash flow needs.</p>
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
                <h3 className="font-semibold text-gray-900 mb-1">Infrastructure</h3>
                <p className="text-sm">Netlify, Render, GitHub Actions, Docker</p>
              </div>
            </div>
          </CollapsibleSection>

          {/* Footer */}
          <div className="text-sm text-gray-400 pt-6 mt-8 border-t border-gray-200 text-center">
            All trade values and projections are independent statistical estimates and are not affiliated with MLB, the MLBPA, or any club. Spot something that looks off? Use the feedback form above.
          </div>

        </div>
      </main>
    </div>
  )
}

export default About