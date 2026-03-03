import { useState, useEffect } from 'react';
import { useProspects } from '../../hooks/useApi';
import ProspectsTable from '../../components/Tables/Prospects/ProspectTable';
import { PROSPECT_YEARS, PROSPECT_DEFAULT_YEAR } from '../../config';


const teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 
               'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'ATH', 
               'PHI', 'PIT', 'SD', 'SF', 'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WSH'];

const hitterPositions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'LF', 'CF', 'RF', 'DH'];

const ProspectsPage = () => {
  const [playerType, setPlayerType] = useState<'hitter' | 'pitcher'>('hitter');
  const [team, setTeam] = useState<string>();
  const [position, setPosition] = useState<string>();
  const [year, setYear] = useState(PROSPECT_DEFAULT_YEAR);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [sortBy, setSortBy] = useState<string>('composite');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  // React Query — replaces manual useState/useEffect/fetch
  const { data, isFetching: loading, error } = useProspects({
    playerType, team, position, year, page, pageSize, sortBy, sortDirection,
  });

  // Reset to first page when filters or sort change
  useEffect(() => { setPage(1); }, [year, playerType, team, position, sortBy, sortDirection]);

  const handleSort = (key: string) => {
    const newDirection = sortBy === key && sortDirection === 'desc' ? 'asc' : 'desc';
    setSortBy(key);
    setSortDirection(newDirection);
  };



  return (
    <div className="min-h-screen bg-gradient-to-br from-surface-900 via-surface-800 to-surface-900">
      <div className="max-w-7xl mx-auto py-16 px-4">
        <h1 className="text-4xl font-bold mb-8 text-white tracking-tight">Prospect Rankings & Values</h1>
        
        <div className="rounded-xl p-6 border border-white/[0.06] bg-surface-800/50 mb-8">
          <div className="flex flex-wrap gap-4 items-center">
            <select 
              value={year}
              onChange={(e) => setYear(Number(e.target.value))}
              className="bg-surface-700/50 border border-white/[0.08] rounded-lg px-4 py-2.5 text-surface-300 focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40"
            >
              {PROSPECT_YEARS.map(y => (
                <option key={y} value={y} className="bg-surface-800">{y}</option>
              ))}
            </select>

            <div className="flex space-x-2">
              <button
                onClick={() => {
                  setPlayerType('hitter');
                  setPosition(undefined);
                }}
                className={`px-4 py-2.5 rounded-lg text-sm font-medium ${
                  playerType === 'hitter'
                    ? 'bg-brand-500 text-surface-900'
                    : 'bg-white/[0.04] text-surface-300 hover:bg-white/[0.08] border border-white/[0.06]'
                }`}
              >
                Hitters
              </button>
              <button
                onClick={() => {
                  setPlayerType('pitcher');
                  setPosition(undefined);
                }}
                className={`px-4 py-2.5 rounded-lg text-sm font-medium ${
                  playerType === 'pitcher'
                    ? 'bg-brand-500 text-surface-900'
                    : 'bg-white/[0.04] text-surface-300 hover:bg-white/[0.08] border border-white/[0.06]'
                }`}
              >
                Pitchers
              </button>
            </div>
            
            <select
              value={team || ''}
              onChange={(e) => setTeam(e.target.value || undefined)}
              className="bg-surface-700/50 border border-white/[0.08] rounded-lg px-4 py-2.5 text-surface-300 focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40"
            >
              <option value="" className="bg-surface-800">All Teams</option>
              {teams.map(team => (
                <option key={team} value={team} className="bg-surface-800">{team}</option>
              ))}
            </select>

            {/* Only show position filter for hitters */}
            {playerType === 'hitter' && (
              <select
                value={position || ''}
                onChange={(e) => setPosition(e.target.value || undefined)}
                className="bg-surface-700/50 border border-white/[0.08] rounded-lg px-4 py-2.5 text-surface-300 focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40"
              >
                <option value="" className="bg-surface-800">All Positions</option>
                {hitterPositions.map(pos => (
                  <option key={pos} value={pos} className="bg-surface-800">{pos}</option>
                ))}
              </select>
            )}
  
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
            {error instanceof Error ? error.message : 'Failed to fetch prospects'}
          </div>
        )}
  
        {data && (
          <div>
            <div className="text-sm text-surface-400 mb-4">
              Found {data.count} prospects
            </div>
            <div className="rounded-xl border border-white/[0.06] bg-surface-800/50">
              <ProspectsTable 
                data={data}
                playerType={playerType}
                year={year}
                currentPage={page}
                totalPages={Math.ceil(data.count / pageSize)}
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

export default ProspectsPage;