import React, { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { filterPlayers } from '../../services/api'

interface PlayerResult {
  id: number;
  name: string;
  team: string;
  war: number;
  position: string;
}

const PlayerSearch = () => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<PlayerResult[]>([])
  const navigate = useNavigate()

  const handleSearch = async (value: string) => {
    console.log('Search value:', value);  // Debug input
    setQuery(value);
    
    if (value.length < 2) {
        console.log('Search too short');  // Debug early return
        setResults([]);
        return;
    }

    try {
      console.log('Calling API with:', value);  // Debug API call
      const response = await filterPlayers({ year: 2025, search: value });
      console.log('API Response:', response);  // Debug response
      
      if (!response || !response.players) {
          console.error('Invalid response structure');
          return;
      }

      setResults(response.players.map(p => ({
          id: p.id,
          name: p.name,
          team: p.team,
          position: p.position,
          war: p.war
      })));
  } catch (err) {
      console.error('Search failed:', err);
  }
};

  const handleSelect = (player: PlayerResult) => {
    navigate(`/player/${player.id}`)
    setQuery('')
    setResults([])
  }

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
                {player.war != null ? player.war.toFixed(1) : '-'} WAR
            </span>
        </button>
          ))}
        </div>
      )}
    </div>
  )
}

export default PlayerSearch