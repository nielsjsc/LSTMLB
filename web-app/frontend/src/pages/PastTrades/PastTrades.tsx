import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  getPastTrades,
  PastTradeSummary,
  TradeSideSummary,
} from '../../services/api';
import { getTeamColors } from '../../utils/teamColors';

// ── Helpers ──────────────────────────────────────────────────────────────────

const fmtDate = (d: string) => {
  const dt = new Date(d + 'T12:00:00');
  return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

const fmtMoney = (n: number) => {
  const abs = Math.abs(n);
  if (abs >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `$${(n / 1_000).toFixed(0)}K`;
  return `$${n}`;
};

const MLB_TEAMS = [
  'ARI','ATL','BAL','BOS','CHC','CHW','CIN','CLE','COL','DET',
  'HOU','KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','OAK',
  'PHI','PIT','SDP','SFG','SEA','STL','TBR','TEX','TOR','WSN',
];

const YEARS = Array.from({ length: 12 }, (_, i) => 2025 - i); // 2025..2014

// ── Side summary component ──────────────────────────────────────────────────

function TradeSide({
  side,
  isWinner,
  isProjected,
}: {
  side: TradeSideSummary;
  isWinner: boolean;
  isProjected: boolean;
}) {
  const colors = getTeamColors(side.team);
  // For projected trades, use projected values; for actual, use actual
  const displayWar = isProjected ? (side.projected_total_war ?? 0) : side.total_war;
  const displaySurplus = isProjected ? (side.projected_total_surplus ?? 0) : side.total_surplus;

  return (
    <div className="flex-1 min-w-0">
      {/* Team header */}
      <div className="flex items-center gap-2 mb-2">
        <div
          className="w-2.5 h-2.5 rounded-full flex-shrink-0"
          style={{ backgroundColor: colors.primary }}
        />
        <span className="text-sm font-semibold text-surface-100 truncate">
          {side.team_name}
        </span>
        {isWinner && (
          <span className="text-[10px] font-bold uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-500/15 text-emerald-400 flex-shrink-0">
            W
          </span>
        )}
      </div>

      {/* Received players */}
      <div className="space-y-0.5 mb-2">
        {side.players_received.map((p) => {
          const pWar = isProjected && p.has_projection ? (p.projected_war ?? 0) : p.war_with_team;
          return (
            <div key={p.mlb_id} className="flex items-center gap-1.5 text-xs">
              <Link
                to={`/players/${p.mlb_id}`}
                className="text-blue-400 hover:text-blue-300 truncate"
              >
                {p.name}
              </Link>
              {pWar > 0 && (
                <span className="text-surface-400 flex-shrink-0">
                  {pWar} WAR
                </span>
              )}
              {isProjected && !p.has_projection && (
                <span className="text-surface-600 flex-shrink-0 italic text-[10px]">
                  no proj.
                </span>
              )}
              {p.prospect_fv && (
                <span className="text-amber-400/70 flex-shrink-0">
                  FV {p.prospect_fv}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Side stats */}
      <div className="flex items-center gap-3 text-[11px] text-surface-500">
        <span>{displayWar} WAR{isProjected ? ' (proj.)' : ''}</span>
        <span>{fmtMoney(displaySurplus)} surplus{isProjected ? ' (proj.)' : ''}</span>
      </div>
    </div>
  );
}

// ── Trade card ──────────────────────────────────────────────────────────────

function TradeCard({ trade }: { trade: PastTradeSummary }) {
  const winnerColors = getTeamColors(trade.winner);
  const isProjected = trade.evaluation_type === 'projected';
  const displaySurplus = isProjected
    ? (trade.projected_surplus_diff ?? trade.surplus_diff)
    : trade.surplus_diff;

  return (
    <Link
      to={`/trades/${trade.trade_id}`}
      className="block rounded-xl border border-white/[0.06] bg-surface-800/50 hover:bg-surface-800/80 transition-colors p-4"
    >
      {/* Header row */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <span className="text-xs text-surface-400">{fmtDate(trade.date)}</span>
          {isProjected && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/15 font-medium">
              Projected
            </span>
          )}
          {trade.has_cash && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400/70 border border-amber-500/10">
              + Cash
            </span>
          )}
          {trade.has_ptbnl && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-surface-700 text-surface-400 border border-white/[0.06]">
              + PTBNL
            </span>
          )}
        </div>

        {/* Surplus difference badge */}
        <div className="flex items-center gap-1.5 text-xs">
          <div
            className="w-1.5 h-1.5 rounded-full"
            style={{ backgroundColor: winnerColors.primary }}
          />
          <span className="text-surface-300 font-medium">
            {trade.winner}
          </span>
          <span className={isProjected ? 'text-blue-400/70' : 'text-emerald-400/70'}>
            +{fmtMoney(displaySurplus)}
          </span>
        </div>
      </div>

      {/* Two sides */}
      <div className="flex gap-4">
        {trade.sides.map((side, idx) => (
          <div key={side.team} className="contents">
            {idx > 0 && (
              <div className="flex items-center self-stretch">
                <div className="w-px h-full bg-white/[0.06]" />
              </div>
            )}
            <TradeSide
              side={side}
              isWinner={side.team === trade.winner}
              isProjected={isProjected}
            />
          </div>
        ))}
      </div>
    </Link>
  );
}

// ── Main page ───────────────────────────────────────────────────────────────

export default function PastTrades() {
  const [trades, setTrades] = useState<PastTradeSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(true);

  // Filters & sorting
  const [sortBy, setSortBy] = useState('date');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [teamFilter, setTeamFilter] = useState('');
  const [yearFilter, setYearFilter] = useState<number | ''>('');
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');

  const pageSize = 25;

  const fetchTrades = useCallback(async () => {
    setLoading(true);
    try {
      const res = await getPastTrades({
        page,
        page_size: pageSize,
        sort_by: sortBy,
        sort_dir: sortDir,
        team: teamFilter || undefined,
        year: yearFilter || undefined,
        search: search || undefined,
      });
      setTrades(res.trades);
      setTotal(res.total);
      setTotalPages(res.total_pages);
    } catch (e) {
      console.error('Failed to load trades:', e);
    } finally {
      setLoading(false);
    }
  }, [page, sortBy, sortDir, teamFilter, yearFilter, search]);

  useEffect(() => {
    fetchTrades();
  }, [fetchTrades]);

  // Reset page when filters change
  useEffect(() => {
    setPage(1);
  }, [sortBy, sortDir, teamFilter, yearFilter, search]);

  const handleSearch = () => {
    setSearch(searchInput);
  };

  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortBy(field);
      setSortDir('desc');
    }
  };

  const sortBtnClass = (field: string) =>
    `px-3 py-1.5 rounded-lg text-xs font-medium transition-colors ${
      sortBy === field
        ? 'bg-blue-500/15 text-blue-400 border border-blue-500/25'
        : 'bg-surface-800 text-surface-400 border border-white/[0.06] hover:bg-surface-700'
    }`;

  return (
    <div className="max-w-6xl mx-auto px-4 py-8">
      {/* Page header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-surface-100 mb-1">Past Trades</h1>
        <p className="text-sm text-surface-500">
          {total.toLocaleString()} evaluated trades (2014-2025) &mdash; recent offseason trades show projected values
        </p>
      </div>

      {/* Filters bar */}
      <div className="flex flex-wrap items-center gap-3 mb-6">
        {/* Search */}
        <div className="flex items-center gap-1">
          <input
            type="text"
            placeholder="Search players or descriptions..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            className="w-64 px-3 py-1.5 rounded-lg bg-surface-800 border border-white/[0.06] text-sm text-surface-200 placeholder-surface-500 focus:outline-none focus:border-blue-500/40"
          />
          <button
            onClick={handleSearch}
            className="px-3 py-1.5 rounded-lg bg-surface-800 border border-white/[0.06] text-surface-400 text-xs hover:bg-surface-700"
          >
            Search
          </button>
        </div>

        {/* Team filter */}
        <select
          value={teamFilter}
          onChange={(e) => setTeamFilter(e.target.value)}
          className="px-3 py-1.5 rounded-lg bg-surface-800 border border-white/[0.06] text-sm text-surface-300 focus:outline-none"
        >
          <option value="">All Teams</option>
          {MLB_TEAMS.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>

        {/* Year filter */}
        <select
          value={yearFilter}
          onChange={(e) => setYearFilter(e.target.value ? Number(e.target.value) : '')}
          className="px-3 py-1.5 rounded-lg bg-surface-800 border border-white/[0.06] text-sm text-surface-300 focus:outline-none"
        >
          <option value="">All Years</option>
          {YEARS.map((y) => (
            <option key={y} value={y}>{y}</option>
          ))}
        </select>

        {/* Sort buttons */}
        <div className="flex items-center gap-1.5 ml-auto">
          <span className="text-xs text-surface-500 mr-1">Sort:</span>
          <button onClick={() => handleSort('date')} className={sortBtnClass('date')}>
            Date {sortBy === 'date' && (sortDir === 'desc' ? '\u2193' : '\u2191')}
          </button>
          <button onClick={() => handleSort('surplus_diff')} className={sortBtnClass('surplus_diff')}>
            Surplus {sortBy === 'surplus_diff' && (sortDir === 'desc' ? '\u2193' : '\u2191')}
          </button>
          <button onClick={() => handleSort('total_trade_war')} className={sortBtnClass('total_trade_war')}>
            Total WAR {sortBy === 'total_trade_war' && (sortDir === 'desc' ? '\u2193' : '\u2191')}
          </button>
          <button onClick={() => handleSort('max_prospect_fv')} className={sortBtnClass('max_prospect_fv')}>
            Top FV {sortBy === 'max_prospect_fv' && (sortDir === 'desc' ? '\u2193' : '\u2191')}
          </button>
        </div>
      </div>

      {/* Trade list */}
      {loading ? (
        <div className="flex justify-center py-20">
          <div className="w-6 h-6 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
        </div>
      ) : trades.length === 0 ? (
        <div className="text-center py-20 text-surface-500">
          No trades found matching your filters.
        </div>
      ) : (
        <div className="space-y-3">
          {trades.map((trade) => (
            <TradeCard key={trade.trade_id} trade={trade} />
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-center gap-2 mt-8">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="px-3 py-1.5 rounded-lg bg-surface-800 border border-white/[0.06] text-sm text-surface-300 disabled:opacity-30 hover:bg-surface-700"
          >
            Previous
          </button>
          <span className="text-sm text-surface-400">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="px-3 py-1.5 rounded-lg bg-surface-800 border border-white/[0.06] text-sm text-surface-300 disabled:opacity-30 hover:bg-surface-700"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
