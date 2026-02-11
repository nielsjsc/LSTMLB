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
      <div className="relative">
        <svg className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-surface-500 pointer-events-none" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          type="text"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          placeholder="Search players..."
          className="w-full pl-9 pr-3 py-2 rounded-lg text-sm bg-white/[0.06] border border-white/[0.08] text-surface-200 placeholder-surface-500 
                     focus:outline-none focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40 focus:bg-white/[0.08]
                     transition-all"
        />
      </div>
      {results.length > 0 && (
        <div className="absolute z-50 w-full mt-1.5 bg-surface-800 border border-white/[0.08] rounded-xl shadow-xl shadow-black/40 overflow-hidden max-h-80 overflow-y-auto">
          {results.map((player) => (
            <button
              key={player.id}
              onClick={() => handleSelect(player)}
              className="w-full px-4 py-2.5 text-left hover:bg-white/[0.06] flex justify-between items-center gap-3 transition-colors border-b border-white/[0.04] last:border-b-0"
            >
              <div className="flex items-center gap-2 min-w-0">
                <span className="text-sm font-medium text-white truncate">{player.name}</span>
                <span className="text-xs text-surface-400 shrink-0">
                  {player.team.toUpperCase()} · {player.position}
                </span>
              </div>
              <span className="text-xs font-mono text-brand-400 shrink-0">
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