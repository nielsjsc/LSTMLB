import React, { useState, useEffect } from 'react';
import { getProspects, ProspectResponse } from '../../services/api';
import ProspectsTable from '../../components/Tables/Prospects/ProspectTable';


const teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 
               'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'ATH', 
               'PHI', 'PIT', 'SD', 'SF', 'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WSH'];

const hitterPositions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'LF', 'CF', 'RF', 'DH'];

// Update years array to show most recent first
const years = [2025, 2024, 2023, 2022];

const ProspectsPage = () => {
  const [playerType, setPlayerType] = useState<'hitter' | 'pitcher'>('hitter');
  const [team, setTeam] = useState<string>();
  const [position, setPosition] = useState<string>();
  const [year, setYear] = useState(2025);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(50);  // Changed to 50 to match Projections
  const [data, setData] = useState<ProspectResponse>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();
  const [sortBy, setSortBy] = useState<string>('composite');  // Default sort
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');

  const fetchProspects = async (newPage?: number) => {
    try {
      setLoading(true);
      setError(undefined);
      const currentPage = newPage || page;
      
      const response = await getProspects(
        playerType,
        team,
        position,
        year,
        currentPage,
        pageSize,
        sortBy,
        sortDirection
      );
      
      setData(response);
      if (newPage) setPage(newPage);
    } catch (error) {
      console.error('Failed to fetch prospects:', error);
      setError('Failed to fetch prospects');
    } finally {
      setLoading(false);
    }
  };

  const handleSort = (key: string) => {
    const newDirection = sortBy === key && sortDirection === 'desc' ? 'asc' : 'desc';
    setSortBy(key);
    setSortDirection(newDirection);
    fetchProspects(1);
  };

  // Update useEffect to match Projections behavior
  useEffect(() => {
    fetchProspects(1);
  }, [year, playerType, team, position, sortBy, sortDirection]);



  const handlePageChange = async (newPage: number) => {
    try {
      setLoading(true);
      setError(undefined);
      const response = await getProspects(
        playerType,
        team,
        position,
        year,
        newPage,
        pageSize,
        sortBy,  // Now this is string | undefined
        sortDirection
      );
      setData(response);
      setPage(newPage);
    } catch (error) {
      setError('Failed to fetch prospects');
    } finally {
      setLoading(false);
    }
  };



  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto py-16 px-4">
        <h1 className="text-4xl font-bold mb-8 text-white">Prospect Rankings & Values</h1>
        
        <div className="rounded-xl p-6 border border-slate-700/50 bg-slate-800/50 mb-8">
          <div className="flex flex-wrap gap-4 items-center">
            <select 
              value={year}
              onChange={(e) => setYear(Number(e.target.value))}
              className="bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 text-gray-300"
            >
              {years.map(y => (
                <option key={y} value={y} className="bg-slate-800">{y}</option>
              ))}
            </select>

            <div className="flex space-x-2">
              <button
                onClick={() => {
                  setPlayerType('hitter');
                  setPosition(undefined);
                }}
                className={`px-4 py-3 rounded-lg font-medium ${
                  playerType === 'hitter'
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-700/50 text-gray-300 hover:bg-slate-600/50'
                }`}
              >
                Hitters
              </button>
              <button
                onClick={() => {
                  setPlayerType('pitcher');
                  setPosition(undefined);
                }}
                className={`px-4 py-3 rounded-lg font-medium ${
                  playerType === 'pitcher'
                    ? 'bg-emerald-600 text-white'
                    : 'bg-slate-700/50 text-gray-300 hover:bg-slate-600/50'
                }`}
              >
                Pitchers
              </button>
            </div>
            
            <select
              value={team || ''}
              onChange={(e) => setTeam(e.target.value || undefined)}
              className="bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 text-gray-300"
            >
              <option value="" className="bg-slate-800">All Teams</option>
              {teams.map(team => (
                <option key={team} value={team} className="bg-slate-800">{team}</option>
              ))}
            </select>

            {/* Only show position filter for hitters */}
            {playerType === 'hitter' && (
              <select
                value={position || ''}
                onChange={(e) => setPosition(e.target.value || undefined)}
                className="bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 text-gray-300"
              >
                <option value="" className="bg-slate-800">All Positions</option>
                {hitterPositions.map(pos => (
                  <option key={pos} value={pos} className="bg-slate-800">{pos}</option>
                ))}
              </select>
            )}
  
            {loading && (
              <div className="text-emerald-400 flex items-center">
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
            <div className="text-sm text-gray-400 mb-4">
              Found {data.count} prospects
            </div>
            <div className="rounded-xl border border-slate-700/50 bg-slate-800/50">
              <ProspectsTable 
                data={data}
                playerType={playerType}
                year={year}
                currentPage={page}
                totalPages={Math.ceil(data.count / pageSize)}
                onPageChange={(newPage) => handlePageChange(newPage)}
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