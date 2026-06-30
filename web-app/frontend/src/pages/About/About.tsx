import React, { useState } from 'react'
import { FaGithub } from 'react-icons/fa'

const Section = ({ title, children }: { title: string; children: React.ReactNode }) => (
  <section className="mb-14">
    <h2 className="text-xl font-bold text-gray-900 font-display tracking-tight mb-4 pb-2 border-b border-gray-200">{title}</h2>
    <div className="space-y-4 text-[15px] text-gray-600 leading-relaxed">{children}</div>
  </section>
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
      // Replace YOUR_FORM_ID below with your Formspree form ID
      // Sign up at formspree.io, create a form, and paste the ID here
      const res = await fetch('https://formspree.io/f/YOUR_FORM_ID', {
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
        <p className="text-green-800 font-medium text-sm">Thanks for the feedback — it's genuinely appreciated.</p>
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
          placeholder="What's on your mind? Specific players you think are over/undervalued, methodology questions, things that seem off — anything goes."
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
          <div className="mb-14">
            <h1 className="font-display text-4xl font-extrabold text-gray-900 tracking-tight mb-3">About LongBall</h1>
            <p className="text-lg text-gray-500 mb-6">
              An open-source platform for quantifying MLB trade value through long-term player projections, contract analysis, and surplus value modeling.
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

          {/* Motivation */}
          <Section title="Motivation">
            <p>
              Trade value is one of the most discussed yet least quantified concepts in baseball. Every deadline deal prompts debate about who "won" a trade, but there is no publicly available, systematic framework for assigning dollar values to players on the trade market. This project attempts to build one.
            </p>
            <p>
              Computing trade value requires two things: projected future on-field production (WAR) and projected future salary obligations. The difference between the two — surplus value — captures how much value a team receives above what it pays. But surplus alone does not fully explain trade behavior. A player with 3 projected WAR and $5M in surplus is not equivalent to a player with 1 projected WAR and the same surplus. Scarcity, roster concentration, and certainty all matter. This project models that distinction.
            </p>
          </Section>

          {/* Projection Models */}
          <Section title="Projection Models">
            <p>
              Player projections are generated by a hybrid system that pairs a bidirectional LSTM neural network for hitting with Marcel-style statistical models for pitching, defense, and baserunning. Each method was chosen based on where it performs best: the LSTM excels at learning nonlinear sequential patterns in high-signal batting data, while Marcel's weighted-average approach is more reliable for noisier, lower-signal metrics where deep learning tends to overfit.
            </p>

            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mt-6 mb-2">Hitting (LSTM)</h3>
            <p>
              The batter model uses a two-stage transfer learning approach. Stage one pre-trains on classical features (BB%, K%, AVG, OBP, SLG, wOBA, and per-150 counting stats) using data back to 1950. Stage two fine-tunes on the Statcast era (2015+), expanding the feature set to include expected stats (xwOBA, xBA, xSLG) and batted-ball metrics (EV50, sweet-spot rate). LSTM layers are frozen during fine-tuning so only the output projection adapts to the new features, preserving the temporal patterns learned from 75 years of data.
            </p>
            <p>
              Rather than predicting absolute stat lines, the model predicts year-over-year deltas from the player's last observed season. This prevents the autoregressive loop from regressing every player toward league average over long horizons. Counting stats (HR, 2B, 3B, RBI, R) are derived from the model's predicted wOBA scaled by each player's career counting-stat profile, preserving individual power and speed signatures.
            </p>

            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mt-6 mb-2">Pitching (Marcel)</h3>
            <p>
              Pitching projections use a Marcel-style weighted average rather than an LSTM. The Marcel method takes up to three recent seasons of rate statistics, weights them 5/4/3 by recency (further scaled by sample size), and regresses toward league average. Year-over-year projections are then generated by applying empirically-derived aging curves.
            </p>
            <p>
              After computing the Marcel base, a set of multivariate equations nudge rate stats (K%, BB%, HR/FB, BABIP) based on more predictive peripheral indicators — including Stuff+, Location+, and Pitching+ for pitchers with Statcast data. ERA and FIP are reconstructed from component rates rather than predicted directly, producing internally consistent projections. ERA is derived from reconstructed FIP plus a James-Stein shrunk career ERA-FIP gap. Separate equations are applied for starters and relievers.
            </p>

            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mt-6 mb-2">Defense and Baserunning (Marcel)</h3>
            <p>
              Defensive and baserunning projections also use the Marcel framework. Defensive stats are modeled using Statcast fielding run values (total runs, range runs, arm runs, and double-play runs for infielders; framing, throwing, and blocking runs for catchers). Each stat has its own regression amount calibrated from empirical year-to-year stabilization analysis — for example, infield arm runs are so noisy (year-to-year r ≈ −0.16) that projections carry very low reliability weight, while catcher blocking runs stabilize much faster.
            </p>
            <p>
              After regression, a reliability multiplier is applied based on out-of-sample grid search across 2016–2025 Statcast data, further shrinking projections that carry weak predictive signal. Aging curves then apply position-specific decline rates, with minimum decline floors enforced for older players to correct for small-sample artifacts in the empirical curves.
            </p>
          </Section>

          {/* Data Pipeline */}
          <Section title="Data Pipeline">
            <p>
              All statistics are normalized to rate form before model input. Batting stats are converted to per-150 games, pitcher stats to per-batter-faced, and defensive stats to per-1350 innings. This allows the models to learn performance level independent of playing time, which is projected separately.
            </p>
            <p>
              A reliability regression module applies James-Stein shrinkage to small samples before prediction. Each stat has a research-based stabilization point (e.g., K% stabilizes at 60 PA, wOBA at 200 PA, BABIP at 1200 batters faced). Players below these thresholds are regressed toward a recency-weighted career prior blended with the league average. Park factors, sourced from FanGraphs five-year averages, neutralize stats on input and re-apply them when calculating WAR.
            </p>
            <p>
              Empirical aging curves are derived from within-player year-over-year deltas using the paired difference method, with a correction for survivorship bias — the tendency for poor performers to exit the dataset at higher rates, which causes raw age curves to understate true decline.
            </p>
          </Section>

          {/* WAR Calculation */}
          <Section title="WAR Calculation">
            <p>
              Position player WAR is computed from batting runs (using FanGraphs wOBA weights and park-adjusted wRC+), baserunning runs, defensive runs, and positional adjustments, divided by a runs-per-win conversion factor. Pitcher WAR uses FIP-based methodology with a dynamic runs-per-win denominator — an ace who pitches deep into games suppresses the overall run environment, making each run saved more valuable in terms of wins.
            </p>
            <p>
              Players are projected at 150 games for position players (135 for catchers), 32 starts for starting pitchers, and 65 innings for relievers. While injuries and playing time are significant components of player value, they are difficult to predict reliably and are not currently modeled.
            </p>
          </Section>

          {/* Trade Value */}
          <Section title="From WAR to Trade Value">
            <p>
              WAR is converted to dollars using a convex power-law model: <span className="font-mono text-sm text-gray-700">$8.59M × WAR<sup>1.18</sup></span>, with 4% annual inflation from a 2025 base. The exponent greater than one captures the superlinear pricing of high-WAR players — a five-win player commands substantially more on the trade market than five one-win players. The parameters were calibrated by minimizing median absolute trade imbalance across 744 MLB trades from 2014 to 2024.
            </p>
            <p>
              Trade value is then the sum of projected WAR dollar values over a player's remaining team-control years, minus projected salary obligations. Pre-arbitration players are valued at $720K per year. Arbitration salaries are estimated as a percentage of market value (15% for Arb-1, 25% for Arb-2, 40% for Arb-3). Contract options (player, team, vesting, and opt-outs) are evaluated and incorporated into the timeline.
            </p>
          </Section>

          {/* Prospect Valuations */}
          <Section title="Prospect Valuations">
            <p>
              Prospects without significant MLB track records are valued using FanGraphs FV grades and consensus ranking data. Base values range from $1M (30 FV) to $120M (70 FV), with a top-100 ranking multiplier that scales from 1.5x at rank 1 down to 1.0x at rank 100. As prospects accumulate MLB playing time, their valuation transitions linearly from the FV-based estimate to the projection-based estimate over the first 300 games (batters), 45 starts (SP), or 65 appearances (RP).
            </p>
            <p>
              A separate MiLB projection model uses minor league rate statistics with level-adjusted priors (accounting for the empirical translation rates between levels) to estimate future MLB performance for players still in the minor league system.
            </p>
          </Section>

          {/* Bayesian Model */}
          <Section title="Bayesian Aging Model">
            <p>
              An independent Bayesian hierarchical model, implemented in NumPyro, provides a second opinion on aging trajectories. It estimates population-level aging curve parameters (peak age, decline rate) and player-specific deviations using a three-level hierarchy. PA-weighting in the likelihood ensures that a 600-PA season constrains a player's talent estimate far more tightly than a 100-PA season, without arbitrary cutoffs.
            </p>
            <p>
              The model is fit in two stages: population aging parameters are learned from 1950–2020 data to avoid leakage, then current player projections use only 2021–2025 data with the learned curve as a prior.
            </p>
          </Section>

          {/* Limitations */}
          <Section title="Limitations">
            <p>
              Long-range player projections are inherently uncertain. No model can predict injuries, role changes, mechanical adjustments, or the dozens of other factors that alter a career trajectory. The projections presented here represent a statistical expectation conditional on continued health and opportunity — they are not predictions of what will happen, but estimates of baseline expected value.
            </p>
            <p>
              Playing time is assumed rather than predicted. The WAR-to-dollar model, while calibrated on real trades, reflects average market behavior and cannot capture the idiosyncratic preferences of individual front offices. Prospect valuations rely on consensus scouting grades that update annually and can shift substantially between seasons.
            </p>
            <p>
              Pitching projections in particular should be treated with caution. The Marcel methodology is more stable than the earlier LSTM approach, but pitching metrics remain noisier than hitting, and small-sample pitchers are aggressively regressed toward league average. Long-range pitcher values in the trade simulator will reflect this conservatism.
            </p>
          </Section>

          {/* Tech Stack */}
          <section className="mb-14">
            <h2 className="text-xl font-bold text-gray-900 font-display tracking-tight mb-4 pb-2 border-b border-gray-200">Technical Stack</h2>
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-x-8 gap-y-2 text-[15px] text-gray-600">
              <div><span className="text-gray-400">Models</span> <span className="ml-2">PyTorch, NumPyro</span></div>
              <div><span className="text-gray-400">Backend</span> <span className="ml-2">FastAPI, SQLAlchemy</span></div>
              <div><span className="text-gray-400">Frontend</span> <span className="ml-2">React, TypeScript</span></div>
              <div><span className="text-gray-400">Data</span> <span className="ml-2">Statcast, FanGraphs, Spotrac</span></div>
              <div><span className="text-gray-400">Database</span> <span className="ml-2">SQLite</span></div>
              <div><span className="text-gray-400">Styling</span> <span className="ml-2">Tailwind CSS</span></div>
            </div>
          </section>

          {/* Feedback */}
          <section className="mb-14">
            <h2 className="text-xl font-bold text-gray-900 font-display tracking-tight mb-2 pb-2 border-b border-gray-200">Feedback</h2>
            <p className="text-[15px] text-gray-500 mb-6">
              Spotted a projection that seems way off? Have a methodology question, bug report, or feature idea? I'd love to hear it.
            </p>
            <FeedbackForm />
          </section>

          {/* Footer */}
          <div className="text-sm text-gray-400 pt-6 border-t border-gray-200">
            All models and data pipelines are open source under the MIT license. Contributions and feedback are welcome on GitHub.
          </div>

        </div>
      </main>
    </div>
  )
}

export default About
