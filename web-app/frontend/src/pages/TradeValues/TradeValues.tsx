import  { useState, useEffect } from 'react';
import { getTradeValueRankings, TradeValueRankingsResponse } from '../../services/api';
import TradeValuesTable from '../../components/Tables/TradeValues/TradeValuesTable';

const teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 
               'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'ATH', 
               'PHI', 'PIT', 'SD', 'SF', 'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WSH', 'FA'];

const TradeValues = () => {
  const [team, setTeam] = useState<string>();
  const [data, setData] = useState<TradeValueRankingsResponse>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [sortBy, setSortBy] = useState<string>('trade_value');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  const fetchTradeValues = async (newPage?: number) => {
    try {
      setLoading(true);
      setError(undefined);
      const currentPage = newPage || page;
      
      const response = await getTradeValueRankings({
        team,
        page: currentPage,
        pageSize,
        sortBy,
        sortDirection
      });
      
      setData(response);
      if (newPage) setPage(newPage);
    } catch (error) {
      console.error('Failed to fetch trade values:', error);
      setError('Failed to fetch trade values');
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (key: string) => {
    const newDirection = sortBy === key && sortDirection === 'desc' ? 'asc' : 'desc';
    setSortBy(key);
    setSortDirection(newDirection);
    fetchTradeValues(1);
  };

  useEffect(() => {
    fetchTradeValues(1);
  }, [team, sortBy, sortDirection]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-surface-900 via-surface-800 to-surface-900">
      <div className="max-w-7xl mx-auto py-16 px-4">
        <h1 className="text-4xl font-bold mb-8 text-white tracking-tight">Trade Value Rankings</h1>
        
        <div className="rounded-xl p-6 border border-white/[0.06] bg-surface-800/50 mb-8">
          <div className="flex flex-wrap gap-4">
            <select
              value={team || ''}
              onChange={(e) => setTeam(e.target.value || undefined)}
              className="bg-surface-700/50 border border-white/[0.08] rounded-lg px-4 py-2.5 text-surface-300 focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40"
            >
              <option value="" className="bg-surface-800">All Teams</option>
              {teams.map(t => (
                <option key={t} value={t} className="bg-slate-800">{t}</option>
              ))}
            </select>

            {loading && (
              <div className="text-brand-400 flex items-center">
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
            {error}
          </div>
        )}

        {data && (
          <div>
            <div className="text-sm text-surface-400 mb-4">
              Found {data.total_count} players
            </div>
            <div className="rounded-xl border border-white/[0.06] bg-surface-800/50">
              <TradeValuesTable 
                data={data}
                currentPage={page}
                totalPages={data.total_pages}
                onPageChange={(newPage) => fetchTradeValues(newPage)}
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