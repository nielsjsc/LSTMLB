import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { filterPlayers } from '../../services/api'
import { CURRENT_YEAR } from '../../config'

interface PlayerResult {
  id: number;
  name: string;
  team: string;
  position: string;
  war_bat?: number | null;
  war_pit?: number | null;
}

const PlayerSearch = () => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<PlayerResult[]>([])
  const navigate = useNavigate()

  const formatWAR = (player: PlayerResult) => {
    // If player has both hitting and pitching WAR, sum them
    if (player.war_bat != null && player.war_pit != null) {
      return (player.war_bat + player.war_pit).toFixed(1);
    }
    // Otherwise show whichever WAR value exists
    return (player.war_bat ?? player.war_pit)?.toFixed(1) ?? '-';
  };

  const handleSearch = async (value: string) => {
    setQuery(value);
    
    if (value.length < 2) {
      setResults([]);
      return;
    }

    try {
      const response = await filterPlayers({ year: CURRENT_YEAR, search: value });
      
      if (!response || !response.players) {
        console.error('Invalid response structure');
        return;
      }

      // Just use the backend results directly
      setResults(response.players.map(p => ({
        id: p.real_id,
        name: p.name,
        team: p.team,
        position: p.position,
        war_bat: p.war_bat,
        war_pit: p.war_pit
      })));
    } catch (err) {
      console.error('Search failed:', err);
    }
  };

  const handleSelect = (player: PlayerResult) => {
    setQuery('');
    setResults([]);
    navigate(`/players/${player.id}`);
  };

  return (
    <div className="relative">
      <input
        type="text"
        value={query}
        onChange={(e) => handleSearch(e.target.value)}
        placeholder="Search players..."
        className="w-full p-2 border rounded text-black bg-white"
      />
      {results.length > 0 && (
        <div className="absolute z-10 w-full mt-1 bg-white border rounded shadow-lg">
          {results.map((player) => (
            <button
              key={player.id}
              onClick={() => handleSelect(player)}
              className="w-full p-2 text-left text-gray-800 hover:bg-gray-100 flex justify-between items-center"
            >
              <span>{player.name} ({player.team.toUpperCase()} - {player.position})</span>
              <span className="text-gray-600">
                {formatWAR(player)} WAR
              </span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
};

export default PlayerSearch;