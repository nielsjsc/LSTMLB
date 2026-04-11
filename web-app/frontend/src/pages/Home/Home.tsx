import React, { useState, useEffect } from 'react'
import { Link } from 'react-router-dom'
import { FaGithub } from 'react-icons/fa'
import { getTradeValueRankings, getPastTrades, type TradeValueRankings, type PastTradeSummary } from '../../services/api'

const fmtDollar = (v: number) => {
  if (Math.abs(v) >= 1e9) return `$${(v / 1e9).toFixed(1)}B`
  if (Math.abs(v) >= 1e6) return `$${(v / 1e6).toFixed(0)}M`
  if (Math.abs(v) >= 1e3) return `$${(v / 1e3).toFixed(0)}K`
  return `$${v.toFixed(0)}`
}

const fmtWar = (w: number) => w >= 0 ? `+${w.toFixed(1)}` : w.toFixed(1)

const fmtDate = (d: string) => {
  const date = new Date(d + 'T00:00:00')
  return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
}

const Home = () => {
  const [topPlayers, setTopPlayers] = useState<TradeValueRankings[]>([])
  const [recentTrades, setRecentTrades] = useState<PastTradeSummary[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    const fetchData = async () => {
      try {
        const [tvRes, tradeRes] = await Promise.all([
          getTradeValueRankings({ pageSize: 10, sortBy: 'trade_value', sortDirection: 'desc' }),
          getPastTrades({ page_size: 5, sort_by: 'total_trade_war', sort_dir: 'desc', min_war: 5 }),
        ])
        if (!cancelled) {
          setTopPlayers(tvRes.players)
          setRecentTrades(tradeRes.trades)
        }
      } catch {
        // Non-critical — page degrades gracefully to empty tables
      } finally {
        if (!cancelled) setLoading(false)
      }
    }
    fetchData()
    return () => { cancelled = true }
  }, [])

  return (
    <div className="min-h-screen bg-[#F5F3EE]">
      <div className="max-w-6xl mx-auto px-4 py-10">
        {/* Hero */}
        <div className="text-center mb-10">
          <h1 className="font-display text-5xl font-extrabold mb-2 bg-clip-text text-transparent bg-gradient-brand tracking-tight">
            LONGBALL
          </h1>
          <p className="text-lg text-gray-500 mb-4">Open Source Baseball Analytics</p>
          <a
            href="https://github.com/nielsjsc/LSTMLB"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 px-4 py-1.5 rounded-md bg-gray-900 hover:bg-gray-800 transition-colors text-white text-sm"
          >
            <FaGithub className="h-4 w-4" />
            <span>GitHub</span>
          </a>
        </div>

        {/* Data Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Top Trade Values — wider column */}
          <div className="lg:col-span-3 bg-white rounded-lg border border-gray-200 shadow-card overflow-hidden">
            <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
              <h2 className="text-sm font-bold text-gray-900 font-display uppercase tracking-wide">Top Trade Values</h2>
              <Link to="/tradevalues" className="text-xs text-accent-blue hover:underline font-medium">View All &rarr;</Link>
            </div>
            {loading ? (
              <div className="px-4 py-8 text-center text-gray-400 text-sm">Loading...</div>
            ) : topPlayers.length === 0 ? (
              <div className="px-4 py-8 text-center text-gray-400 text-sm">No data available</div>
            ) : (
              <table className="w-full text-sm">
                <thead>
                  <tr className="text-[11px] text-gray-500 uppercase tracking-wider border-b border-gray-100">
                    <th className="text-left pl-4 pr-2 py-2 font-medium">#</th>
                    <th className="text-left px-2 py-2 font-medium">Player</th>
                    <th className="text-left px-2 py-2 font-medium hidden sm:table-cell">Pos</th>
                    <th className="text-right px-2 py-2 font-medium">Value</th>
                    <th className="text-right px-2 py-2 font-medium hidden md:table-cell">WAR</th>
                    <th className="text-right pl-2 pr-4 py-2 font-medium hidden md:table-cell">Ctrl</th>
                  </tr>
                </thead>
                <tbody>
                  {topPlayers.map((p, i) => (
                    <tr key={p.real_id} className={`border-b border-gray-50 hover:bg-gray-50 transition-colors ${i % 2 === 1 ? 'bg-[#F7F4EF]' : ''}`}>
                      <td className="pl-4 pr-2 py-2 text-gray-400 font-medium">{i + 1}</td>
                      <td className="px-2 py-2">
                        <Link to={`/players/${p.mlb_id || p.real_id}`} className="text-accent-blue hover:underline font-medium">
                          {p.name}
                        </Link>
                        <span className="text-gray-400 text-xs ml-1.5">{p.team}</span>
                      </td>
                      <td className="px-2 py-2 text-gray-500 hidden sm:table-cell">{p.position}</td>
                      <td className="px-2 py-2 text-right font-semibold text-gray-900">{fmtDollar(p.trade_value)}</td>
                      <td className="px-2 py-2 text-right text-gray-600 hidden md:table-cell">{p.contract_war.toFixed(1)}</td>
                      <td className="pl-2 pr-4 py-2 text-right text-gray-500 hidden md:table-cell">{p.years_control}y</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Quick Links sidebar */}
          <div className="lg:col-span-2 flex flex-col gap-6">
            {/* Navigation cards */}
            <div className="grid grid-cols-2 gap-3">
              <Link to="/projections" className="bg-white rounded-lg border border-gray-200 shadow-card p-4 hover:shadow-card-hover hover:border-gray-300 transition-all group">
                <div className="text-sm font-bold text-gray-900 font-display mb-1">Projections</div>
                <div className="text-xs text-gray-500">ML-powered career forecasts</div>
              </Link>
              <Link to="/tradesimulator" className="bg-white rounded-lg border border-gray-200 shadow-card p-4 hover:shadow-card-hover hover:border-gray-300 transition-all group">
                <div className="text-sm font-bold text-gray-900 font-display mb-1">Trade Sim</div>
                <div className="text-xs text-gray-500">Evaluate trade scenarios</div>
              </Link>
              <Link to="/prospects" className="bg-white rounded-lg border border-gray-200 shadow-card p-4 hover:shadow-card-hover hover:border-gray-300 transition-all group">
                <div className="text-sm font-bold text-gray-900 font-display mb-1">Prospects</div>
                <div className="text-xs text-gray-500">Rankings &amp; valuations</div>
              </Link>
              <Link to="/about" className="bg-white rounded-lg border border-gray-200 shadow-card p-4 hover:shadow-card-hover hover:border-gray-300 transition-all group">
                <div className="text-sm font-bold text-gray-900 font-display mb-1">About</div>
                <div className="text-xs text-gray-500">Methodology &amp; models</div>
              </Link>
            </div>
          </div>
        </div>

        {/* Recent Trades */}
        <div className="mt-6 bg-white rounded-lg border border-gray-200 shadow-card overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3 border-b border-gray-100">
            <h2 className="text-sm font-bold text-gray-900 font-display uppercase tracking-wide">Biggest Trades</h2>
            <Link to="/trades" className="text-xs text-accent-blue hover:underline font-medium">View All &rarr;</Link>
          </div>
          {loading ? (
            <div className="px-4 py-8 text-center text-gray-400 text-sm">Loading...</div>
          ) : recentTrades.length === 0 ? (
            <div className="px-4 py-8 text-center text-gray-400 text-sm">No trades available</div>
          ) : (
            <div className="divide-y divide-gray-100">
              {recentTrades.map((t) => (
                <Link
                  key={t.trade_id}
                  to={`/trades/${t.trade_id}`}
                  className="flex items-center gap-4 px-4 py-3 hover:bg-gray-50 transition-colors"
                >
                  <div className="text-xs text-gray-400 w-20 shrink-0">{fmtDate(t.date)}</div>
                  <div className="flex-1 min-w-0">
                    <div className="text-sm text-gray-900 truncate">{t.description}</div>
                    <div className="text-xs text-gray-500 mt-0.5">
                      {t.n_players} players &middot; {fmtWar(t.total_trade_war)} combined WAR
                    </div>
                  </div>
                  {t.winner && (
                    <div className="text-xs shrink-0">
                      <span className="text-emerald-600 font-medium">{t.winner}</span>
                      <span className="text-gray-300 mx-1">/</span>
                      <span className="text-red-500">{t.loser}</span>
                    </div>
                  )}
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default Home