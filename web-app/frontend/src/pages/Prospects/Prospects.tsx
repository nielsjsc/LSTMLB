import { useState, useEffect } from 'react';
import { useProspects } from '../../hooks/useApi';
import ProspectsTable from '../../components/Tables/Prospects/ProspectTable';
import { PROSPECT_YEARS, PROSPECT_DEFAULT_YEAR } from '../../config';

type ViewMode = 'grades' | 'stats' | 'all_stats';

const teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 
               'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'ATH', 
               'PHI', 'PIT', 'SD', 'SF', 'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WSH'];

const hitterPositions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'LF', 'CF', 'RF', 'DH'];

const viewModeLabels: Record<ViewMode, string> = {
  grades: 'Prospect Grades',
  stats: 'Stats + Grades',
  all_stats: 'All Stats',
};

const ProspectsPage = () => {
  const [playerType, setPlayerType] = useState<'hitter' | 'pitcher'>('hitter');
  const [team, setTeam] = useState<string>();
  const [position, setPosition] = useState<string>();
  const [year, setYear] = useState(PROSPECT_DEFAULT_YEAR);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [sortBy, setSortBy] = useState<string>('composite');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');
  const [viewMode, setViewMode] = useState<ViewMode>('grades');
  const [minPa, setMinPa] = useState<number | undefined>();
  const [minIp, setMinIp] = useState<number | undefined>();

  const { data, isFetching: loading, error } = useProspects({
    playerType, team, position, year, page, pageSize, sortBy, sortDirection,
    view: viewMode,
    minPa: playerType === 'hitter' ? minPa : undefined,
    minIp: playerType === 'pitcher' ? minIp : undefined,
  });

  // Reset to first page when filters or sort change
  useEffect(() => { setPage(1); }, [year, playerType, team, position, sortBy, sortDirection, viewMode, minPa, minIp]);

  const handleSort = (key: string) => {
    const newDirection = sortBy === key && sortDirection === 'desc' ? 'asc' : 'desc';
    setSortBy(key);
    setSortDirection(newDirection);
  };

  return (
    <div className="min-h-screen bg-\[#F7F7F5\]">
      <div className="max-w-[95rem] mx-auto py-6 px-4">
        <h1 className="text-2xl font-bold mb-5 text-gray-900 tracking-tight font-display">Prospect Rankings & Values</h1>
        
        <div className="border-b border-gray-200 pb-4 mb-6">
          {/* Row 1: Main filters */}
          <div className="flex flex-wrap gap-4 items-center mb-4">
            <select 
              value={year}
              onChange={(e) => setYear(Number(e.target.value))}
              className="bg-white border border-gray-200 rounded px-3 py-2 text-gray-600 focus:ring-2 focus:ring-brand-500/25 focus:border-brand-300"
            >
              {PROSPECT_YEARS.map(y => (
                <option key={y} value={y} className="bg-white">{y}</option>
              ))}
            </select>

            <div className="flex space-x-2">
              <button
                onClick={() => { setPlayerType('hitter'); setPosition(undefined); }}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  playerType === 'hitter'
                    ? 'bg-brand-500 text-white'
                    : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200'
                }`}
              >
                Hitters
              </button>
              <button
                onClick={() => { setPlayerType('pitcher'); setPosition(undefined); }}
                className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                  playerType === 'pitcher'
                    ? 'bg-brand-500 text-white'
                    : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200'
                }`}
              >
                Pitchers
              </button>
            </div>
            
            <select
              value={team || ''}
              onChange={(e) => setTeam(e.target.value || undefined)}
              className="bg-white border border-gray-200 rounded px-3 py-2 text-gray-600 focus:ring-2 focus:ring-brand-500/25 focus:border-brand-300"
            >
              <option value="" className="bg-white">All Teams</option>
              {teams.map(t => (
                <option key={t} value={t} className="bg-white">{t}</option>
              ))}
            </select>

            {playerType === 'hitter' && (
              <select
                value={position || ''}
                onChange={(e) => setPosition(e.target.value || undefined)}
                className="bg-white border border-gray-200 rounded px-3 py-2 text-gray-600 focus:ring-2 focus:ring-brand-500/25 focus:border-brand-300"
              >
                <option value="" className="bg-white">All Positions</option>
                {hitterPositions.map(pos => (
                  <option key={pos} value={pos} className="bg-white">{pos}</option>
                ))}
              </select>
            )}

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

          {/* Row 2: View mode tabs + stat filters */}
          <div className="flex flex-wrap gap-4 items-center">
            <div className="flex space-x-1 bg-gray-50 rounded-lg p-1">
              {(Object.keys(viewModeLabels) as ViewMode[]).map((mode) => (
                <button
                  key={mode}
                  onClick={() => setViewMode(mode)}
                  className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
                    viewMode === mode
                      ? 'bg-brand-50 text-brand-500 border border-brand-200'
                      : 'text-gray-500 hover:text-gray-800 hover:bg-gray-50'
                  }`}
                >
                  {viewModeLabels[mode]}
                </button>
              ))}
            </div>

            {/* Stat filters — only show when stats view is active */}
            {viewMode !== 'grades' && (
              <div className="flex items-center gap-3">
                <span className="text-[11px] text-gray-400 uppercase tracking-wider">Min:</span>
                {playerType === 'hitter' ? (
                  <div className="flex items-center gap-1.5">
                    <label className="text-[11px] text-gray-500">PA</label>
                    <input
                      type="number"
                      min="0"
                      value={minPa ?? ''}
                      onChange={(e) => {
                        const v = e.target.value ? Math.max(0, Number(e.target.value)) : undefined;
                        setMinPa(v);
                      }}
                      placeholder="0"
                      className="w-16 bg-white border border-gray-200 rounded px-2 py-1 text-[11px] text-gray-600 focus:ring-1 focus:ring-brand-500/25"
                    />
                  </div>
                ) : (
                  <div className="flex items-center gap-1.5">
                    <label className="text-[11px] text-gray-500">IP</label>
                    <input
                      type="number"
                      min="0"
                      value={minIp ?? ''}
                      onChange={(e) => {
                        const v = e.target.value ? Math.max(0, Number(e.target.value)) : undefined;
                        setMinIp(v);
                      }}
                      placeholder="0"
                      step="0.1"
                      className="w-16 bg-white border border-gray-200 rounded px-2 py-1 text-[11px] text-gray-600 focus:ring-1 focus:ring-brand-500/25"
                    />
                  </div>
                )}
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
            <div className="text-sm text-gray-500 mb-3">
              Found {data.count} prospects
            </div>
            <div className="border-t border-gray-200">
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
                viewMode={viewMode}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default ProspectsPage;