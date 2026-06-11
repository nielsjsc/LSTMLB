import { useState, useEffect } from 'react';
import { useProjections } from '../../hooks/useApi';
import ProjectionsTable from '../../components/Tables/Projections/ProjectionsTable';
import { CURRENT_YEAR } from '../../config';


const years = Array.from(
  { length: 15 }, 
  (_, i) => CURRENT_YEAR + i
);
const teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 
               'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'ATH', 
               'PHI', 'PIT', 'SD', 'SF', 'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WSH', 'FA'];

const hitterPositions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'LF', 'CF', 'RF', 'DH'];
const pitcherPositions = ['SP', 'RP'];



const ProjectionsPage = () => {
  // Filter / pagination state
  const [year, setYear] = useState(CURRENT_YEAR);
  const [playerType, setPlayerType] = useState<'hitter' | 'pitcher'>('hitter');
  const [team, setTeam] = useState<string>();
  const [position, setPosition] = useState<string>();
  const [projectionType, setProjectionType] = useState<'ros' | 'preseason'>('ros');
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);
  const [sortBy, setSortBy] = useState<string>(
    playerType === 'hitter' ? 'war_bat' : 'war_pit'
  );
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // React Query — replaces manual useState/useEffect/fetch
  const { data, isFetching: loading, error } = useProjections({
    year, playerType, team, position, projectionType, page, pageSize, sortBy, sortDirection,
  });

  // Reset to first page when filters or sort change
  useEffect(() => { setPage(1); }, [year, playerType, team, position, projectionType, sortBy, sortDirection]);

  // Reset projection type when year changes (preseason only for current year)
  useEffect(() => {
    if (year !== CURRENT_YEAR) setProjectionType('ros');
  }, [year]);

  const handleSort = (key: string) => {
    const newDirection = sortBy === key && sortDirection === 'desc' ? 'asc' : 'desc';
    setSortBy(key);
    setSortDirection(newDirection);
  };


  return (
    <div className="min-h-screen bg-\[#F5F3EE\]">
      <div className="max-w-7xl mx-auto py-6 px-4">
        <h1 className="text-2xl font-bold mb-5 text-gray-900 tracking-tight font-display">Player Projections</h1>
        
        {/* Controls Section */}
        <div className="border-b border-gray-200 pb-4 mb-6">
          <div className="flex flex-wrap gap-4 items-center">
            <select 
              value={year} 
              onChange={(e) => setYear(Number(e.target.value))}
              className="bg-white border border-gray-200 rounded px-3 py-2.5 text-sm text-gray-600 focus:ring-2 focus:ring-brand-500/25 focus:border-brand-300 min-h-10"
            >
              {years.map(year => (
                <option key={year} value={year} className="bg-white">{year}</option>
              ))}
            </select>
  
            <div className="flex space-x-2">
            <button
              onClick={() => {
                setPlayerType('hitter');
                setPosition(undefined);
                setSortBy('war_bat');  // Set default sort for hitters
                setSortDirection('desc');
              }}
              className={`px-3 py-2.5 min-h-10 min-w-10 rounded text-sm font-medium ${
                playerType === 'hitter'
                  ? 'bg-brand-500 text-white'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200'
              }`}
            >
              Hitters
            </button>
            <button
              onClick={() => {
                setPlayerType('pitcher');
                setPosition(undefined);
                setSortBy('war_pit');  // Set default sort for pitchers
                setSortDirection('desc');
              }}
              className={`px-3 py-2.5 min-h-10 min-w-10 rounded text-sm font-medium ${
                playerType === 'pitcher'
                  ? 'bg-brand-500 text-white'
                  : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200'
              }`}
            >
              Pitchers
            </button>
            </div>

            {year === CURRENT_YEAR && (
              <div className="flex space-x-2">
                <button
                  onClick={() => setProjectionType('ros')}
                  className={`px-3 py-2.5 min-h-10 min-w-10 rounded text-sm font-medium ${
                    projectionType === 'ros'
                      ? 'bg-brand-500 text-white'
                      : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200'
                  }`}
                >
                  ROS
                </button>
                <button
                  onClick={() => setProjectionType('preseason')}
                  className={`px-3 py-2.5 min-h-10 min-w-10 rounded text-sm font-medium ${
                    projectionType === 'preseason'
                      ? 'bg-brand-500 text-white'
                      : 'bg-gray-50 text-gray-600 hover:bg-gray-100 border border-gray-200'
                  }`}
                >
                  Preseason
                </button>
              </div>
            )}
  
            <select
              value={team || ''}
              onChange={(e) => setTeam(e.target.value || undefined)}
              className="bg-white border border-gray-200 rounded px-3 py-2.5 text-sm text-gray-600 focus:ring-2 focus:ring-brand-500/25 focus:border-brand-300 min-h-10"
            >
              <option value="" className="bg-white">All Teams</option>
              {teams.map(team => (
                <option key={team} value={team} className="bg-white">{team}</option>
              ))}
            </select>
  
            <select
              value={position || ''}
              onChange={(e) => setPosition(e.target.value || undefined)}
              className="bg-white border border-gray-200 rounded px-3 py-2.5 text-sm text-gray-600 focus:ring-2 focus:ring-brand-500/25 focus:border-brand-300 min-h-10"
            >
              <option value="" className="bg-white">All Positions</option>
              {(playerType === 'hitter' ? hitterPositions : pitcherPositions).map(pos => (
                <option key={pos} value={pos} className="bg-white">{pos}</option>
              ))}
            </select>
  
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
            {error instanceof Error ? error.message : 'Failed to fetch projections'}
          </div>
        )}
  
        {data && (
          <div>
            <div className="text-sm text-gray-500 mb-3">
              Found {data.total_count} players
            </div>
            <div className="border-t border-gray-200">
              <ProjectionsTable 
                data={data}
                playerType={playerType}
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

export default ProjectionsPage;