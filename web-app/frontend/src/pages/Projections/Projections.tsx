import React, { useState, useEffect } from 'react';
import { getProjections, ProjectionResponse } from '../../services/api';
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
  // Existing state
  const [year, setYear] = useState(CURRENT_YEAR);
  const [playerType, setPlayerType] = useState<'hitter' | 'pitcher'>('hitter');
  const [team, setTeam] = useState<string>();
  const [position, setPosition] = useState<string>();
  const [data, setData] = useState<ProjectionResponse>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  
  // Update pagination state
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);  // Changed from 25 to match backend
  
  // Add sorting state
  const [sortBy, setSortBy] = useState<string>(
    playerType === 'hitter' ? 'war_bat' : 'war_pit'
  );
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  const fetchProjections = async (newPage?: number) => {
    try {
      setLoading(true);
      setError(undefined);
      const currentPage = newPage || page;
      
      const response = await getProjections(
        year, 
        playerType, 
        team, 
        position,
        currentPage,
        pageSize,
        sortBy,
        sortDirection
      );
      
      setData(response);
      if (newPage) setPage(newPage);
    } catch (error) {
      if (error instanceof DOMException && error.name === 'AbortError') return;
      setError('Failed to fetch projections');
    } finally {
      setLoading(false);
    }
  };



  const handleSort = (key: string) => {
    const newDirection = sortBy === key && sortDirection === 'desc' ? 'asc' : 'desc';
    setSortBy(key);
    setSortDirection(newDirection);
    fetchProjections(1);  // Reset to first page when sorting
  };

  // Update useEffect dependencies to include sorting
  useEffect(() => {
    fetchProjections(1);
  }, [year, playerType, team, position, sortBy, sortDirection]);


  return (
    <div className="min-h-screen bg-gradient-to-br from-surface-900 via-surface-800 to-surface-900">
      <div className="max-w-7xl mx-auto py-16 px-4">
        <h1 className="text-4xl font-bold mb-8 text-white tracking-tight">Player Projections</h1>
        
        {/* Controls Section */}
        <div className="rounded-xl p-6 border border-white/[0.06] bg-surface-800/50 mb-8">
          <div className="flex flex-wrap gap-4 items-center">
            <select 
              value={year} 
              onChange={(e) => setYear(Number(e.target.value))}
              className="bg-surface-700/50 border border-white/[0.08] rounded-lg px-4 py-2.5 text-surface-300 focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40"
            >
              {years.map(year => (
                <option key={year} value={year} className="bg-surface-800">{year}</option>
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
                setSortBy('war_pit');  // Set default sort for pitchers
                setSortDirection('desc');
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
  
            <select
              value={position || ''}
              onChange={(e) => setPosition(e.target.value || undefined)}
              className="bg-surface-700/50 border border-white/[0.08] rounded-lg px-4 py-2.5 text-surface-300 focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40"
            >
              <option value="" className="bg-surface-800">All Positions</option>
              {(playerType === 'hitter' ? hitterPositions : pitcherPositions).map(pos => (
                <option key={pos} value={pos} className="bg-surface-800">{pos}</option>
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
              <ProjectionsTable 
                data={data}
                playerType={playerType}
                currentPage={page}
                totalPages={data.total_pages}
                onPageChange={(newPage) => fetchProjections(newPage)}
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