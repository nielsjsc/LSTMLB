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
  if (abs >= 1_000_000) return `$${(abs / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `$${(abs / 1_000).toFixed(0)}K`;
  return `$${abs}`;
};

const MLB_TEAMS = [
  'ARI','ATL','BAL','BOS','CHC','CHW','CIN','CLE','COL','DET',
  'HOU','KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','OAK',
  'PHI','PIT','SDP','SFG','SEA','STL','TBR','TEX','TOR','WSN',
];

const YEARS = Array.from({ length: CURRENT_YEAR - 2013 }, (_, i) => CURRENT_YEAR - 1 - i);

// ── Trade grade based on surplus differential ───────────────────────────────

type TradeGrade = 'A+' | 'A' | 'A-' | 'B+' | 'B' | 'B-' | 'C' | 'F';

function getTradeGrade(surplusDiff: number): TradeGrade {
  const abs = Math.abs(surplusDiff);
  if (abs >= 80_000_000) return 'A+';
  if (abs >= 50_000_000) return 'A';
  if (abs >= 35_000_000) return 'A-';
  if (abs >= 25_000_000) return 'B+';
  if (abs >= 15_000_000) return 'B';
  if (abs >= 8_000_000) return 'B-';
  if (abs >= 3_000_000) return 'C';
  return 'F';
}

const GRADE_STYLES: Record<TradeGrade, { text: string; bg: string; border: string }> = {
  'A+': { text: 'text-emerald-300', bg: 'bg-emerald-500/15', border: 'border-emerald-500/30' },
  'A':  { text: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20' },
  'A-': { text: 'text-emerald-400', bg: 'bg-emerald-500/8',  border: 'border-emerald-500/15' },
  'B+': { text: 'text-sky-400',     bg: 'bg-sky-500/10',     border: 'border-sky-500/20' },
  'B':  { text: 'text-sky-400',     bg: 'bg-sky-500/8',      border: 'border-sky-500/15' },
  'B-': { text: 'text-sky-400/80',  bg: 'bg-sky-500/6',      border: 'border-sky-500/10' },
  'C':  { text: 'text-amber-400',   bg: 'bg-amber-500/10',   border: 'border-amber-500/20' },
  'F':  { text: 'text-surface-500', bg: 'bg-surface-800',    border: 'border-white/[0.06]' },
};

// ── Confidence badge config ─────────────────────────────────────────────────

const CONFIDENCE_CFG: Record<EvaluationConfidence, { label: string; color: string; bg: string; tip: string }> = {
  definitive: { label: 'Final',    color: 'text-emerald-400', bg: 'bg-emerald-500/10', tip: '4+ years of data — clear outcome' },
  maturing:   { label: 'Maturing', color: 'text-amber-400',   bg: 'bg-amber-500/10',   tip: '2-3 years of data — picture forming' },
  early:      { label: 'Early',    color: 'text-sky-400',     bg: 'bg-sky-500/10',     tip: 'Recent trade — early returns only' },
  projected:  { label: 'Too Early',color: 'text-surface-500', bg: 'bg-surface-700',    tip: 'No actual data yet' },
};

// ── Side column in trade card ───────────────────────────────────────────────

function SideColumn({ side, isWinner, showWar }: {
  side: TradeSideSummary;
  isWinner: boolean;
  showWar: boolean;
}) {
  const colors = getTeamColors(side.team);

  return (
    <div className="flex-1 min-w-0 px-4 py-3">
      {/* Team header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5">
          <div className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: colors.primary }} />
          <span className={`text-[12px] font-semibold ${isWinner ? 'text-surface-100' : 'text-surface-400'}`}>
            {side.team}
            {isWinner && <span className="ml-1 text-emerald-400 text-[10px]">✓</span>}
          </span>
        </div>
        {showWar && (
          <span className="text-[11px] font-mono tabular-nums text-surface-500">
            {side.total_war} WAR
          </span>
        )}
      </div>

      {/* Players */}
      <div className="space-y-0.5">
        {side.players_received.slice(0, 4).map((p) => {
          const isPureProspect = !!p.prospect_fv && p.war_with_team === 0;

          return (
            <div key={p.mlb_id} className="flex items-center justify-between gap-1">
              <div className="flex items-center gap-1 min-w-0">
                <span className="text-[11px] text-surface-300 truncate">{p.name}</span>
                {p.prospect_fv && (
                  <span className="text-[9px] text-amber-400/70 shrink-0">FV{p.prospect_fv}</span>
                )}
              </div>
              {showWar && !isPureProspect && p.war_with_team !== 0 && (
                <span className={`text-[10px] font-mono tabular-nums shrink-0 ${p.war_with_team > 0 ? 'text-emerald-400/70' : 'text-red-400/70'}`}>
                  {p.war_with_team > 0 ? '+' : ''}{p.war_with_team}
                </span>
              )}
            </div>
          );
        })}
        {side.players_received.length > 4 && (
          <span className="text-[10px] text-surface-600">+{side.players_received.length - 4} more</span>
        )}
      </div>
    </div>
  );
}

// ── Trade card ──────────────────────────────────────────────────────────────

function TradeCard({ trade }: { trade: PastTradeSummary }) {
  const confidence = (trade.evaluation_confidence || 'definitive') as EvaluationConfidence;
  const showWar = confidence !== 'projected';
  const grade = getTradeGrade(trade.surplus_diff);
  const gs = GRADE_STYLES[grade];
  const cc = CONFIDENCE_CFG[confidence];
  const winnerColors = getTeamColors(trade.winner);
  const loserColors = getTeamColors(trade.loser);
  const sides = trade.sides || [];

  return (
    <Link
      to={`/trades/${trade.trade_id}`}
      className="group block rounded-xl border border-white/[0.06] bg-surface-850/60 hover:bg-surface-800/80 hover:border-white/[0.10] transition-all duration-200"
    >
      {/* Top bar */}
      <div className="flex items-center gap-3 px-5 py-3 border-b border-white/[0.04]">
        {/* Grade badge */}
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center border text-sm font-bold shrink-0 ${gs.text} ${gs.bg} ${gs.border}`}>
          {grade}
        </div>

        {/* Teams & date */}
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 text-[14px] font-semibold text-surface-100">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: winnerColors.primary }} />
              <span>{trade.winner}</span>
            </div>
            <span className="text-surface-600 text-[12px]">vs</span>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: loserColors.primary }} />
              <span>{trade.loser}</span>
            </div>
            {trade.n_teams > 2 && (
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400 font-medium">
                {trade.n_teams}-Team
              </span>
            )}
          </div>
          <div className="flex items-center gap-2 mt-0.5">
            <span className="text-[12px] text-surface-500">{fmtDate(trade.date)}</span>
            <span
              className={`text-[9px] px-1.5 py-px rounded font-medium uppercase tracking-wider ${cc.bg} ${cc.color}`}
              title={cc.tip}
            >
              {cc.label}
            </span>
            {trade.is_featured && (
              <span className="text-[11px] text-amber-400" title="Notable trade">★</span>
            )}
          </div>
        </div>

        {/* Surplus advantage */}
        <div className="text-right shrink-0">
          <div className="text-[10px] uppercase tracking-wider text-surface-500 mb-0.5">{trade.winner} surplus</div>
          <div className="text-[15px] font-bold text-surface-100 tabular-nums">
            {fmtMoney(trade.surplus_diff)}
          </div>
        </div>
      </div>

      {/* Side-by-side player lists */}
      <div className="flex divide-x divide-white/[0.04]">
        {sides.map((side) => (
          <SideColumn
            key={side.team}
            side={side}
            isWinner={side.team === trade.winner}
            showWar={showWar}
          />
        ))}
      </div>
    </Link>
  );
}

// ── Main page ───────────────────────────────────────────────────────────────

export default function PastTrades() {
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState('date');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [teamFilter, setTeamFilter] = useState('');
  const [yearFilter, setYearFilter] = useState<number | ''>('');
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');
  const [featuredOnly, setFeaturedOnly] = useState(true);

  const pageSize = 25;

  const { data: res, isFetching: loading } = usePastTrades({
    page, pageSize, sortBy, sortDir,
    team: teamFilter, year: yearFilter, search,
    featured: featuredOnly || undefined,
  });

  const trades = res?.trades ?? [];
  const total = res?.total ?? 0;
  const totalPages = res?.total_pages ?? 1;

  useEffect(() => { setPage(1); }, [sortBy, sortDir, teamFilter, yearFilter, search, featuredOnly]);

  const handleSearch = () => setSearch(searchInput);

  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortBy(field);
      setSortDir('desc');
    }
  };

  const sortBtnClass = (field: string) =>
    `px-3 py-1.5 rounded-lg text-[12px] font-medium transition-all ${
      sortBy === field
        ? 'bg-white/[0.08] text-surface-100 shadow-sm'
        : 'text-surface-500 hover:text-surface-300 hover:bg-white/[0.03]'
    }`;

  return (
    <div className="max-w-6xl mx-auto px-6 py-10">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-surface-100 tracking-tight">Trade History</h1>
        <p className="text-[14px] text-surface-500 mt-1">
          {total.toLocaleString()} evaluated trades &middot; 2014&ndash;Present
        </p>
      </div>

      {/* Filters */}
      <div className="flex flex-wrap items-center gap-3 mb-6 p-4 rounded-xl bg-surface-850/40 border border-white/[0.04]">
        {/* Search */}
        <div className="relative">
          <input
            type="text"
            placeholder="Search players or teams..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
            className="w-56 pl-3 pr-8 py-2 rounded-lg bg-surface-800 border border-white/[0.06] text-[13px] text-surface-200 placeholder-surface-600 focus:outline-none focus:border-blue-500/30 focus:ring-1 focus:ring-blue-500/20 transition-all"
          />
          <button
            onClick={handleSearch}
            className="absolute right-2 top-1/2 -translate-y-1/2 text-surface-500 hover:text-surface-300 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </button>
        </div>

        {/* Team filter */}
        <select
          value={teamFilter}
          onChange={(e) => setTeamFilter(e.target.value)}
          className="px-3 py-2 rounded-lg bg-surface-800 border border-white/[0.06] text-[13px] text-surface-300 focus:outline-none focus:border-blue-500/30 transition-all"
        >
          <option value="">All Teams</option>
          {MLB_TEAMS.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>

        {/* Year filter */}
        <select
          value={yearFilter}
          onChange={(e) => setYearFilter(e.target.value ? Number(e.target.value) : '')}
          className="px-3 py-2 rounded-lg bg-surface-800 border border-white/[0.06] text-[13px] text-surface-300 focus:outline-none focus:border-blue-500/30 transition-all"
        >
          <option value="">All Years</option>
          {YEARS.map((y) => <option key={y} value={y}>{y}</option>)}
        </select>

        {/* Featured toggle */}
        <button
          onClick={() => setFeaturedOnly((f) => !f)}
          className={`px-3 py-2 rounded-lg border text-[13px] font-medium transition-all ${
            featuredOnly
              ? 'bg-amber-500/12 border-amber-500/25 text-amber-400'
              : 'bg-surface-800 border-white/[0.06] text-surface-500 hover:text-surface-300'
          }`}
        >
          ★ Blockbusters
        </button>

        {/* Sort controls */}
        <div className="flex items-center gap-1 ml-auto">
          <span className="text-[11px] text-surface-600 mr-1">Sort:</span>
          <button onClick={() => handleSort('date')} className={sortBtnClass('date')}>
            Date {sortBy === 'date' && (sortDir === 'desc' ? '↓' : '↑')}
          </button>
          <button onClick={() => handleSort('surplus_diff')} className={sortBtnClass('surplus_diff')}>
            Surplus {sortBy === 'surplus_diff' && (sortDir === 'desc' ? '↓' : '↑')}
          </button>
          <button onClick={() => handleSort('total_trade_war')} className={sortBtnClass('total_trade_war')}>
            WAR {sortBy === 'total_trade_war' && (sortDir === 'desc' ? '↓' : '↑')}
          </button>
        </div>
      </div>

      {/* Trade list */}
      {loading ? (
        <div className="flex justify-center py-24">
          <div className="w-6 h-6 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
        </div>
      ) : trades.length === 0 ? (
        <div className="text-center py-20 text-surface-500 text-sm">
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
        <div className="flex items-center justify-center gap-4 mt-8">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="px-4 py-2 rounded-lg text-[13px] font-medium text-surface-400 hover:text-surface-200 hover:bg-white/[0.04] disabled:opacity-30 transition-all"
          >
            ← Previous
          </button>
          <div className="flex items-center gap-1">
            {Array.from({ length: Math.min(7, totalPages) }, (_, i) => {
              let pageNum: number;
              if (totalPages <= 7) {
                pageNum = i + 1;
              } else if (page <= 4) {
                pageNum = i + 1;
              } else if (page >= totalPages - 3) {
                pageNum = totalPages - 6 + i;
              } else {
                pageNum = page - 3 + i;
              }
              return (
                <button
                  key={pageNum}
                  onClick={() => setPage(pageNum)}
                  className={`w-8 h-8 rounded-lg text-[12px] font-medium transition-all ${
                    page === pageNum
                      ? 'bg-blue-500/15 text-blue-400 border border-blue-500/25'
                      : 'text-surface-500 hover:text-surface-300 hover:bg-white/[0.04]'
                  }`}
                >
                  {pageNum}
                </button>
              );
            })}
          </div>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="px-4 py-2 rounded-lg text-[13px] font-medium text-surface-400 hover:text-surface-200 hover:bg-white/[0.04] disabled:opacity-30 transition-all"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}
