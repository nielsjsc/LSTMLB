import { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  PastTradeSummary,
  TradeSideSummary,
  EvaluationConfidence,
} from '../../services/api';
import { usePastTrades } from '../../hooks/useApi';
import { CURRENT_YEAR } from '../../config';
import { getTeamColors } from '../../utils/teamColors';

// ── Helpers ──────────────────────────────────────────────────────────────────

const fmtDate = (d: string) => {
  const dt = new Date(d + 'T12:00:00');
  return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

const fmtMoney = (n: number) => {
  const abs = Math.abs(n);
  const sign = n < 0 ? '-' : '+';
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(0)}K`;
  return `${sign}$${abs}`;
};

const MLB_TEAMS = [
  'ARI','ATL','BAL','BOS','CHC','CHW','CIN','CLE','COL','DET',
  'HOU','KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','OAK',
  'PHI','PIT','SDP','SFG','SEA','STL','TBR','TEX','TOR','WSN',
];

const YEARS = Array.from({ length: CURRENT_YEAR - 2013 }, (_, i) => CURRENT_YEAR - 1 - i);

// ── Confidence tier display ─────────────────────────────────────────────────

const CONFIDENCE_CONFIG: Record<EvaluationConfidence, { label: string; bg: string; text: string; tooltip: string }> = {
  definitive: { label: 'Settled', bg: 'bg-emerald-500/12', text: 'text-emerald-400', tooltip: '4+ years of data — clear outcome' },
  maturing:   { label: 'Maturing', bg: 'bg-amber-500/12', text: 'text-amber-400', tooltip: '2-3 years of data — picture forming' },
  early:      { label: 'Early', bg: 'bg-sky-500/12', text: 'text-sky-400', tooltip: 'Recent trade — early returns only' },
  projected:  { label: 'Projected', bg: 'bg-purple-500/12', text: 'text-purple-400', tooltip: 'No actual data — model projections' },
};

function ConfidenceBadge({ confidence }: { confidence: EvaluationConfidence }) {
  const cfg = CONFIDENCE_CONFIG[confidence] || CONFIDENCE_CONFIG.definitive;
  return (
    <span
      className={`text-[9px] px-1.5 py-px rounded font-semibold uppercase tracking-wider ${cfg.bg} ${cfg.text}`}
      title={cfg.tooltip}
    >
      {cfg.label}
    </span>
  );
}

// ── Helpers ──────────────────────────────────────────────────────────────────

const fmtDate = (d: string) => {
  const dt = new Date(d + 'T12:00:00');
  return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

const fmtMoney = (n: number) => {
  const abs = Math.abs(n);
  const sign = n < 0 ? '-' : '+';
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(0)}K`;
  return `${sign}$${abs}`;
};

const MLB_TEAMS = [
  'ARI','ATL','BAL','BOS','CHC','CHW','CIN','CLE','COL','DET',
  'HOU','KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','OAK',
  'PHI','PIT','SDP','SFG','SEA','STL','TBR','TEX','TOR','WSN',
];

const YEARS = Array.from({ length: CURRENT_YEAR - 2013 }, (_, i) => CURRENT_YEAR - 1 - i);

// ── Compact side summary ────────────────────────────────────────────────────

function SideCompact({ side, isWinner, isProjected }: {
  side: TradeSideSummary;
  isWinner: boolean;
  isProjected: boolean;
}) {
  const colors = getTeamColors(side.team);
  const displayWar = isProjected ? (side.projected_total_war ?? 0) : side.total_war;

  return (
    <div className="flex-1 min-w-0">
      <div className="flex items-center gap-2 mb-1.5">
        <div
          className="w-2 h-2 rounded-full flex-shrink-0"
          style={{ backgroundColor: colors.primary }}
        />
        <span className={`text-[13px] font-semibold truncate ${isWinner ? 'text-surface-100' : 'text-surface-300'}`}>
          {side.team}
        </span>
        {isWinner && (
          <span className="text-[9px] font-bold uppercase tracking-wider px-1 py-px rounded bg-emerald-500/15 text-emerald-400">
            W
          </span>
        )}
      </div>
      <div className="space-y-px pl-4">
        {side.players_received.slice(0, 4).map((p) => {
          const pWar = isProjected && p.has_projection ? (p.projected_war ?? 0) : p.war_with_team;
          const isPureProspect = !!p.prospect_fv && p.seasons_with_team === 0 && p.war_with_team === 0;
          const noData = p.has_data === false;

          // Determine link target
          const prospectLink = p.prospect_id ? `/prospects/${p.prospect_id}` : null;
          const mlbLink = !isPureProspect ? `/players/${p.mlb_id}` : null;
          const linkTo = prospectLink || mlbLink;

          return (
            <div key={p.mlb_id} className="flex items-center gap-1.5">
              {linkTo ? (
                <Link
                  to={linkTo}
                  className="text-[12px] text-surface-300 hover:text-blue-400 truncate transition-colors"
                >
                  {p.name}
                </Link>
              ) : (
                <span className="text-[12px] text-surface-400 truncate">
                  {p.name}
                </span>
              )}
              {noData ? (
                <span className="text-[10px] text-surface-600 italic shrink-0">No Data</span>
              ) : (
                <>
                  {!isPureProspect && pWar !== 0 && (
                    <span className={`text-[11px] flex-shrink-0 font-mono tabular-nums ${isProjected ? 'text-purple-400/60' : 'text-surface-500'}`}>
                      {pWar > 0 ? '+' : ''}{pWar}
                    </span>
                  )}
                </>
              )}
              {p.prospect_fv && (
                <span className="text-[10px] text-amber-400/60 shrink-0">
                  FV{p.prospect_fv}
                </span>
              )}
            </div>
          );
        })}
        {side.players_received.length > 4 && (
          <span className="text-[11px] text-surface-600">+{side.players_received.length - 4} more</span>
        )}
      </div>
      <div className={`pl-4 mt-1 text-[11px] tabular-nums font-mono ${isProjected ? 'text-purple-400/70' : 'text-surface-500'}`}>
        {isProjected ? `${displayWar} proj. WAR` : `${displayWar} WAR`}
      </div>
    </div>
  );
}

// ── Trade row ────────────────────────────────────────────────────────────────

function TradeRow({ trade }: { trade: PastTradeSummary }) {
  const winnerColors = getTeamColors(trade.winner);
  const isProjected = trade.evaluation_type === 'projected';
  const confidence = trade.evaluation_confidence || 'definitive';
  const displaySurplus = isProjected
    ? (trade.projected_surplus_diff ?? trade.surplus_diff)
    : trade.surplus_diff;

  const surplusColor = confidence === 'projected' ? 'text-purple-400'
    : confidence === 'early' ? 'text-sky-400'
    : confidence === 'maturing' ? 'text-amber-400'
    : 'text-emerald-400';

  return (
    <Link
      to={`/trades/${trade.trade_id}`}
      className="group flex items-start gap-4 px-5 py-3.5 hover:bg-white/[0.03] transition-colors border-b border-white/[0.04] last:border-b-0"
    >
      {/* Date column */}
      <div className="w-24 flex-shrink-0 pt-0.5">
        <span className="text-[12px] text-surface-500 tabular-nums">{fmtDate(trade.date)}</span>
        <div className="flex items-center gap-1 mt-0.5">
          {confidence !== 'definitive' && (
            <ConfidenceBadge confidence={confidence} />
          )}
          {trade.is_featured && (
            <span className="text-[10px] text-amber-400" title="Notable trade">★</span>
          )}
          {trade.has_cash && (
            <span className="text-[9px] px-1 py-px rounded bg-amber-500/8 text-amber-400/50">$</span>
          )}
        </div>
      </div>

      {/* Sides */}
      <div className="flex-1 flex gap-6 min-w-0">
        {trade.sides.map((side, idx) => (
          <div key={side.team} className="contents">
            {idx > 0 && (
              <div className="flex items-center self-stretch px-1">
                <div className="w-px h-full bg-white/[0.06]" />
              </div>
            )}
            <SideCompact
              side={side}
              isWinner={side.team === trade.winner}
              isProjected={isProjected}
            />
          </div>
        ))}
      </div>

      {/* Surplus result */}
      <div className="w-28 flex-shrink-0 text-right pt-0.5">
        <div className="flex items-center justify-end gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: winnerColors.primary }} />
          <span className="text-[12px] font-medium text-surface-300">{trade.winner}</span>
        </div>
        <span className={`text-[13px] font-semibold tabular-nums ${surplusColor}`}>
          {fmtMoney(displaySurplus)}
        </span>
      </div>
    </Link>
  );
}

// ── Main page ───────────────────────────────────────────────────────────────

export default function PastTrades() {
  // Filters & sorting
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState('date');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [teamFilter, setTeamFilter] = useState('');
  const [yearFilter, setYearFilter] = useState<number | ''>('');
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [featuredOnly, setFeaturedOnly] = useState(false);

  const pageSize = 25;

  // React Query — replaces useCallback/useEffect/fetch
  const { data: res, isFetching: loading } = usePastTrades({
    page, pageSize, sortBy, sortDir,
    team: teamFilter, year: yearFilter, search,
    featured: featuredOnly || undefined,
  });

  const trades = res?.trades ?? [];
  const total = res?.total ?? 0;
  const totalPages = res?.total_pages ?? 1;

  // Reset to first page when filters/sort change
  useEffect(() => { setPage(1); }, [sortBy, sortDir, teamFilter, yearFilter, search, featuredOnly]);

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
    `px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
      sortBy === field
        ? 'bg-white/[0.08] text-surface-100'
        : 'text-surface-500 hover:text-surface-300'
    }`;

  return (
    <div className="max-w-5xl mx-auto px-4 py-8">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-xl font-bold text-surface-100 tracking-tight">Past Trades</h1>
        <p className="text-[13px] text-surface-500 mt-0.5">
          {total.toLocaleString()} evaluated trades, 2014 &ndash; {CURRENT_YEAR - 1}
        </p>
      </div>

      {/* Controls row */}
      <div className="flex flex-wrap items-center gap-2 mb-4">
        <div className="flex items-center gap-1">
          <input
            type="text"
            placeholder="Search players..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            className="w-52 px-2.5 py-1.5 rounded bg-surface-800 border border-white/[0.06] text-[12px] text-surface-200 placeholder-surface-600 focus:outline-none focus:border-white/[0.12]"
          />
          <button
            onClick={handleSearch}
            className="px-2 py-1.5 rounded bg-surface-800 border border-white/[0.06] text-surface-500 text-[11px] hover:text-surface-300 transition-colors"
          >
            Go
          </button>
        </div>

        <select
          value={teamFilter}
          onChange={(e) => setTeamFilter(e.target.value)}
          className="px-2.5 py-1.5 rounded bg-surface-800 border border-white/[0.06] text-[12px] text-surface-300 focus:outline-none"
        >
          <option value="">All Teams</option>
          {MLB_TEAMS.map((t) => (
            <option key={t} value={t}>{t}</option>
          ))}
        </select>

        <select
          value={yearFilter}
          onChange={(e) => setYearFilter(e.target.value ? Number(e.target.value) : '')}
          className="px-2.5 py-1.5 rounded bg-surface-800 border border-white/[0.06] text-[12px] text-surface-300 focus:outline-none"
        >
          <option value="">All Years</option>
          {YEARS.map((y) => (
            <option key={y} value={y}>{y}</option>
          ))}
        </select>

        <button
          onClick={() => setFeaturedOnly((f) => !f)}
          className={`px-2.5 py-1.5 rounded border text-[12px] font-medium transition-colors ${
            featuredOnly
              ? 'bg-amber-500/15 border-amber-500/30 text-amber-400'
              : 'bg-surface-800 border-white/[0.06] text-surface-500 hover:text-surface-300'
          }`}
          title="Show only notable trades (blockbusters, top prospects, high WAR)"
        >
          ★ Featured
        </button>

        <div className="flex items-center gap-0.5 ml-auto">
          <button onClick={() => handleSort('date')} className={sortBtnClass('date')}>
            Date {sortBy === 'date' && (sortDir === 'desc' ? '\u2193' : '\u2191')}
          </button>
          <button onClick={() => handleSort('surplus_diff')} className={sortBtnClass('surplus_diff')}>
            Surplus {sortBy === 'surplus_diff' && (sortDir === 'desc' ? '\u2193' : '\u2191')}
          </button>
          <button onClick={() => handleSort('total_trade_war')} className={sortBtnClass('total_trade_war')}>
            WAR {sortBy === 'total_trade_war' && (sortDir === 'desc' ? '\u2193' : '\u2191')}
          </button>
          <button onClick={() => handleSort('max_prospect_fv')} className={sortBtnClass('max_prospect_fv')}>
            FV {sortBy === 'max_prospect_fv' && (sortDir === 'desc' ? '\u2193' : '\u2191')}
          </button>
        </div>
      </div>

      {/* Trade list */}
      {loading ? (
        <div className="flex justify-center py-20">
          <div className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
        </div>
      ) : trades.length === 0 ? (
        <div className="text-center py-16 text-surface-500 text-sm">
          No trades found.
        </div>
      ) : (
        <div className="rounded-lg border border-white/[0.06] bg-surface-850/50 overflow-hidden">
          {trades.map((trade) => (
            <TradeRow key={trade.trade_id} trade={trade} />
          ))}
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-6">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="px-3 py-1.5 rounded text-[12px] text-surface-400 hover:text-surface-200 disabled:opacity-30 transition-colors"
          >
            &larr; Previous
          </button>
          <span className="text-[12px] text-surface-500 tabular-nums">
            {page} / {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="px-3 py-1.5 rounded text-[12px] text-surface-400 hover:text-surface-200 disabled:opacity-30 transition-colors"
          >
            Next &rarr;
          </button>
        </div>
      )}
    </div>
  );
}
