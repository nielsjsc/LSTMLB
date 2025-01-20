import React, { useState } from 'react';
import { getProjections, ProjectionResponse } from '../../services/api';
import ProjectionsTable from '../../components/Tables/Projections/ProjectionsTable';
const years = Array.from(
  { length: 15 }, 
  (_, i) => 2025 + i
);
const teams = ['ARI', 'ATL', 'BAL', 'BOS', 'CHC', 'CHW', 'CIN', 'CLE', 'COL', 'DET', 
               'HOU', 'KC', 'LAA', 'LAD', 'MIA', 'MIL', 'MIN', 'NYM', 'NYY', 'ATH', 
               'PHI', 'PIT', 'SD', 'SF', 'SEA', 'STL', 'TB', 'TEX', 'TOR', 'WSH'];

const hitterPositions = ['C', '1B', '2B', '3B', 'SS', 'OF', 'LF', 'CF', 'RF', 'DH'];
const pitcherPositions = ['SP', 'RP'];

const matchesPosition = (playerPosition: string, filterPosition: string): boolean => {
  // Split multi-position players
  const positions = playerPosition.split('/');
  
  // OF filter should match any outfield position
  if (filterPosition === 'OF') {
    return positions.some(pos => ['OF', 'LF', 'CF', 'RF'].includes(pos));
  }
  
  return positions.includes(filterPosition);
};

const ProjectionsPage = () => {
  const [year, setYear] = useState(2025);
  const [playerType, setPlayerType] = useState<'hitter' | 'pitcher'>('hitter');
  const [team, setTeam] = useState<string>();
  const [position, setPosition] = useState<string>();
  const [data, setData] = useState<ProjectionResponse>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string>();

  const handleSearch = async () => {
    try {
      setLoading(true);
      setError(undefined);
      const response = await getProjections(year, playerType, team);
      
      // Filter multi-positions client-side
      if (position && response.players) {
        response.players = response.players.filter(player => 
          matchesPosition(player.position, position)
        );
        response.count = response.players.length;
      }
      
      setData(response);
    } catch (error) {
      console.error('Failed to fetch projections:', error);
      setError('Failed to fetch projections');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto py-8 px-4">
      <h1 className="text-3xl font-bold mb-8">Player Projections</h1>
      <div className="flex flex-wrap gap-4 mb-8">
        <select 
          value={year} 
          onChange={(e) => setYear(Number(e.target.value))}
          className="rounded border p-2"
        >
          {years.map(year => (
            <option key={year} value={year}>{year}</option>
          ))}
        </select>

        <select 
          value={playerType} 
          onChange={(e) => {
            setPlayerType(e.target.value as 'hitter' | 'pitcher');
            setPosition(undefined);
          }}
          className="rounded border p-2"
        >
          <option value="hitter">Hitters</option>
          <option value="pitcher">Pitchers</option>
        </select>

        <select
          value={team || ''}
          onChange={(e) => setTeam(e.target.value || undefined)}
          className="rounded border p-2"
        >
          <option value="">All Teams</option>
          {teams.map(team => (
            <option key={team} value={team}>{team}</option>
          ))}
        </select>

        <select
          value={position || ''}
          onChange={(e) => setPosition(e.target.value || undefined)}
          className="rounded border p-2"
        >
          <option value="">All Positions</option>
          {(playerType === 'hitter' ? hitterPositions : pitcherPositions).map(pos => (
            <option key={pos} value={pos}>{pos}</option>
          ))}
        </select>

        <button
          onClick={handleSearch}
          disabled={loading}
          className={`px-4 py-2 rounded text-white ${
            loading ? 'bg-blue-300' : 'bg-blue-500 hover:bg-blue-600'
          }`}
        >
          {loading ? 'Loading...' : 'Search'}
        </button>
      </div>

      {error && (
        <div className="text-red-500 mb-4">{error}</div>
      )}

      {data && (
        <div>
          <div className="text-sm text-gray-500 mb-4">
            Found {data.count} players
          </div>
          <ProjectionsTable data={data} playerType={playerType} />
        </div>
      )}
    </div>
  );
};

export default ProjectionsPage;