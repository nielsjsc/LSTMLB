import { useState, useEffect } from 'react';
import { useTradeValues } from '../../hooks/useApi';
import TradeValuesTable from '../../components/Tables/TradeValues/TradeValuesTable';

const teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 
               'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'ATH', 
               'PHI', 'PIT', 'SD', 'SF', 'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WSH', 'FA'];

const positions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'LF', 'CF', 'RF', 'DH', 'SP', 'RP'];

const TradeValues = () => {
  const [team, setTeam] = useState<string>();
  const [position, setPosition] = useState<string>();
  const [ageOperator, setAgeOperator] = useState<'>=' | '<=' | '='>('=');
  const [ageValue, setAgeValue] = useState<number>();
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [sortBy, setSortBy] = useState<string>('trade_value');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // React Query — replaces manual useState/useEffect/fetch
  const { data, isFetching: loading, error } = useTradeValues({
    team, position, ageOperator, ageValue, page, pageSize, sortBy, sortDirection,
  });

  // Reset to first page when filters or sort change
  useEffect(() => { setPage(1); }, [team, position, ageOperator, ageValue, sortBy, sortDirection]);

  const handleSort = (key: string) => {
    const newDirection = sortBy === key && sortDirection === 'desc' ? 'asc' : 'desc';
    setSortBy(key);
    setSortDirection(newDirection);
  };

  return (
    <div className="min-h-screen bg-\[#F5F3EE\]">
      <div className="max-w-7xl mx-auto py-6 px-4">
        <h1 className="text-lg font-bold mb-4 text-gray-900 tracking-tight font-display">Trade Value Rankings</h1>
        
        <div className="border-b border-gray-200 pb-4 mb-6">
          <div className="flex flex-wrap gap-4">
            <select
              value={team || ''}
              onChange={(e) => setTeam(e.target.value || undefined)}
              className="bg-white border border-gray-200 rounded px-3 py-2.5 text-sm text-gray-600 focus:ring-2 focus:ring-brand-500/25 focus:border-brand-300 min-h-10"
            >
              <option value="" className="bg-white">All Teams</option>
              {teams.map(t => (
                <option key={t} value={t} className="bg-white">{t}</option>
              ))}
            </select>

            <select
              value={position || ''}
              onChange={(e) => setPosition(e.target.value || undefined)}
              className="bg-white border border-gray-200 rounded px-3 py-2.5 text-sm text-gray-600 focus:ring-2 focus:ring-brand-500/25 focus:border-brand-300 min-h-10"
            >
              <option value="" className="bg-white">All Positions</option>
              {positions.map(pos => (
                <option key={pos} value={pos} className="bg-white">{pos}</option>
              ))}
            </select>

            <div className="flex gap-2 items-center">
              <select
                value={ageOperator}
                onChange={(e) => setAgeOperator(e.target.value as '>=' | '<=' | '=')}
                className="bg-white border border-gray-200 rounded px-2 py-2.5 text-sm text-gray-600 focus:ring-2 focus:ring-brand-500/25 focus:border-brand-300 min-h-10"
              >
                <option value=">=" className="bg-white">Age ≥</option>
                <option value="<=" className="bg-white">Age ≤</option>
                <option value="=" className="bg-white">Age =</option>
              </select>
              <input
                type="number"
                min="18"
                max="50"
                value={ageValue ?? ''}
                onChange={(e) => setAgeValue(e.target.value ? parseInt(e.target.value) : undefined)}
                placeholder="Age"
                className="bg-white border border-gray-200 rounded px-3 py-2.5 text-sm text-gray-600 focus:ring-2 focus:ring-brand-500/25 focus:border-brand-300 min-h-10 w-20"
              />
            </div>

            {loading && (
              <div className="text-brand-500 flex items-center">
                <svg className="animate-spin h-5 w-5 mr-2" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                </svg>
                Loading...
              </div>
            )}
          </div>
        </div>

        {error && (
          <div className="text-red-400 mb-4 rounded-lg px-4 py-2 border border-red-500/20">
            {error instanceof Error ? error.message : 'Failed to fetch trade values'}
          </div>
        )}

        {data && (
          <div>
            <div className="text-xs text-gray-500 mb-2">
              Found {data.total_count} players
            </div>
            <div className="border-t border-gray-200">
              <TradeValuesTable 
                data={data}
                currentPage={page}
                totalPages={data.total_pages}
                onPageChange={setPage}
                onSort={handleSort}
                sortBy={sortBy}
                sortDirection={sortDirection}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default TradeValues;