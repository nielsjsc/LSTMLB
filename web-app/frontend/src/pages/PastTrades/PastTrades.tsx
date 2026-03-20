import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { PastTradeSummary } from '../../services/api';
import { usePastTrades } from '../../hooks/useApi';
import { CURRENT_YEAR } from '../../config';

// ── Helpers ──────────────────────────────────────────────────────────────────

const fmtDate = (d: string) => {
  const dt = new Date(d + 'T12:00:00');
  return dt.toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' });
};

const MLB_TEAMS = [
  'ARI','ATL','BAL','BOS','CHC','CHW','CIN','CLE','COL','DET',
  'HOU','KCR','LAA','LAD','MIA','MIL','MIN','NYM','NYY','OAK',
  'PHI','PIT','SDP','SFG','SEA','STL','TBR','TEX','TOR','WSN',
];

const YEARS = Array.from({ length: CURRENT_YEAR - 2013 }, (_, i) => CURRENT_YEAR - 1 - i);

// ── Sortable column header ──────────────────────────────────────────────────

function SortTh({ label, field, sortBy, sortDir, onSort, className = '' }: {
  label: string;
  field: string;
  sortBy: string;
  sortDir: 'asc' | 'desc';
  onSort: (f: string) => void;
  className?: string;
}) {
  const active = sortBy === field;
  return (
    <th
      onClick={() => onSort(field)}
      className={`cursor-pointer select-none text-left text-[11px] font-medium uppercase tracking-wider px-3 py-2.5 transition-colors ${
        active ? 'text-gray-800' : 'text-gray-400 hover:text-gray-600'
      } ${className}`}
    >
      {label}
      {active && <span className="ml-1">{sortDir === 'desc' ? '↓' : '↑'}</span>}
    </th>
  );
}

// ── Trade row ───────────────────────────────────────────────────────────────

function TradeRow({ trade, onClick }: { trade: PastTradeSummary; onClick: () => void }) {
  const sides = trade.sides || [];
  const side1 = sides[0];
  const side2 = sides[1];

  // Build compact player list: top 3 names
  const playerList = (side: typeof side1) => {
    if (!side) return '';
    const names = side.players_received.slice(0, 3).map((p) => p.name);
    const suffix = side.players_received.length > 3 ? ` +${side.players_received.length - 3}` : '';
    return names.join(', ') + suffix;
  };

  return (
    <tr
      onClick={onClick}
      className="border-b border-gray-100 hover:bg-white/[0.02] transition-colors cursor-pointer"
    >
        {/* Date */}
        <td className="px-3 py-2.5 text-[12px] text-gray-400 tabular-nums whitespace-nowrap">
          {fmtDate(trade.date)}
        </td>

        {/* Teams */}
        <td className="px-3 py-2.5 text-[12px] whitespace-nowrap">
          <span className="text-gray-800 font-medium">{side1?.team || '—'}</span>
          <span className="text-gray-400 mx-1.5">↔</span>
          <span className="text-gray-800 font-medium">{side2?.team || '—'}</span>
          {trade.n_teams > 2 && (
            <span className="text-gray-400 text-[10px] ml-1">+{trade.n_teams - 2}</span>
          )}
        </td>

        {/* Side 1 players */}
        <td className="px-3 py-2.5 text-[12px] text-gray-500 truncate max-w-[200px]">
          {playerList(side1)}
        </td>

        {/* Side 2 players */}
        <td className="px-3 py-2.5 text-[12px] text-gray-500 truncate max-w-[200px]">
          {playerList(side2)}
        </td>

        {/* WAR per side */}
        <td className="px-3 py-2.5 text-[12px] tabular-nums whitespace-nowrap text-right">
          <span className={side1 && side1.total_war >= (side2?.total_war ?? 0) ? 'text-gray-800 font-medium' : 'text-gray-500'}>
            {side1?.total_war ?? '—'}
          </span>
          <span className="text-gray-400 mx-1">/</span>
          <span className={side2 && side2.total_war >= (side1?.total_war ?? 0) ? 'text-gray-800 font-medium' : 'text-gray-500'}>
            {side2?.total_war ?? '—'}
          </span>
        </td>
      </tr>
  );
}

// ── Main page ───────────────────────────────────────────────────────────────

export default function PastTrades() {
  const navigate = useNavigate();
  const [page, setPage] = useState(1);
  const [sortBy, setSortBy] = useState('date');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');
  const [teamFilter, setTeamFilter] = useState('');
  const [yearFilter, setYearFilter] = useState<number | ''>('');
  const [search, setSearch] = useState('');
  const [searchInput, setSearchInput] = useState('');

  const pageSize = 40;

  const { data: res, isFetching: loading } = usePastTrades({
    page, pageSize, sortBy, sortDir,
    team: teamFilter, year: yearFilter, search,
  });

  const trades = res?.trades ?? [];
  const total = res?.total ?? 0;
  const totalPages = res?.total_pages ?? 1;

  useEffect(() => { setPage(1); }, [sortBy, sortDir, teamFilter, yearFilter, search]);

  const handleSearch = () => setSearch(searchInput);

  const handleSort = (field: string) => {
    if (sortBy === field) {
      setSortDir((d) => (d === 'desc' ? 'asc' : 'desc'));
    } else {
      setSortBy(field);
      setSortDir('desc');
    }
  };

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      {/* Header + filters */}
      <div className="flex flex-wrap items-end justify-between gap-4 mb-5">
        <div>
          <h1 className="text-xl font-bold text-gray-900 tracking-tight font-display">Trade History</h1>
          <p className="text-[13px] text-gray-400 mt-0.5">
            {total.toLocaleString()} trades &middot; 2014&ndash;Present
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="relative">
            <input
              type="text"
              placeholder="Search..."
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && handleSearch()}
              onBlur={handleSearch}
              className="w-44 pl-3 pr-7 py-1.5 rounded bg-white border border-gray-200 text-[12px] text-gray-800 placeholder-gray-400 focus:outline-none focus:border-gray-300 transition-colors"
            />
            <svg className="w-3.5 h-3.5 absolute right-2 top-1/2 -translate-y-1/2 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
          </div>

          <select
            value={teamFilter}
            onChange={(e) => setTeamFilter(e.target.value)}
            className="px-2.5 py-1.5 rounded bg-white border border-gray-200 text-[12px] text-gray-600 focus:outline-none"
          >
            <option value="">All Teams</option>
            {MLB_TEAMS.map((t) => <option key={t} value={t}>{t}</option>)}
          </select>

          <select
            value={yearFilter}
            onChange={(e) => setYearFilter(e.target.value ? Number(e.target.value) : '')}
            className="px-2.5 py-1.5 rounded bg-white border border-gray-200 text-[12px] text-gray-600 focus:outline-none"
          >
            <option value="">All Years</option>
            {YEARS.map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
        </div>
      </div>

      {/* Table */}
      {loading ? (
        <div className="flex justify-center py-20">
          <div className="w-5 h-5 border-2 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
        </div>
      ) : trades.length === 0 ? (
        <div className="text-center py-16 text-gray-400 text-sm">
          No trades found.
        </div>
      ) : (
        <div className="border-t border-gray-200 overflow-hidden">
          <table className="w-full">
            <thead>
              <tr className="border-b border-gray-200 bg-gray-50">
                <SortTh label="Date" field="date" sortBy={sortBy} sortDir={sortDir} onSort={handleSort} className="w-24" />
                <th className="text-left text-[11px] font-medium uppercase tracking-wider text-gray-400 px-3 py-2.5 w-36">Teams</th>
                <th className="text-left text-[11px] font-medium uppercase tracking-wider text-gray-400 px-3 py-2.5">Side 1</th>
                <th className="text-left text-[11px] font-medium uppercase tracking-wider text-gray-400 px-3 py-2.5">Side 2</th>
                <SortTh label="WAR" field="total_trade_war" sortBy={sortBy} sortDir={sortDir} onSort={handleSort} className="text-right w-28" />
              </tr>
            </thead>
            <tbody>
              {trades.map((trade) => (
                <TradeRow key={trade.trade_id} trade={trade} onClick={() => navigate(`/trades/${trade.trade_id}`)} />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between mt-5">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="px-3 py-1.5 rounded text-[12px] text-gray-500 hover:text-gray-800 disabled:opacity-30 transition-colors"
          >
            ← Previous
          </button>
          <span className="text-[12px] text-gray-400 tabular-nums">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="px-3 py-1.5 rounded text-[12px] text-gray-500 hover:text-gray-800 disabled:opacity-30 transition-colors"
          >
            Next →
          </button>
        </div>
      )}
    </div>
  );
}
