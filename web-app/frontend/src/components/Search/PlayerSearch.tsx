import React, { useState, useRef, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { filterPlayers } from '../../services/api'
import { CURRENT_YEAR } from '../../config'

interface PlayerResult {
  id: number;
  mlb_id: number | null;
  name: string;
  team: string;
  position: string;
  war_bat?: number | null;
  war_pit?: number | null;
  is_historical?: boolean;
  career_war?: number;
  first_year?: number;
  last_year?: number;
}

const PlayerSearch = () => {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<PlayerResult[]>([])
  const [focusedIdx, setFocusedIdx] = useState(-1)
  const navigate = useNavigate()
  const containerRef = useRef<HTMLDivElement>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Cleanup debounce timer on unmount
  useEffect(() => {
    return () => { if (debounceRef.current) clearTimeout(debounceRef.current); };
  }, []);

  // Close dropdown on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setResults([]);
      }
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  const handleSearch = (value: string) => {
    setQuery(value);
    setFocusedIdx(-1);

    // Clear pending debounce
    if (debounceRef.current) clearTimeout(debounceRef.current);

    if (value.length < 2) {
      setResults([]);
      return;
    }

    // Debounce API call by 250ms to avoid firing on every keystroke
    debounceRef.current = setTimeout(async () => {
      try {
        const response = await filterPlayers({ year: CURRENT_YEAR, search: value });

        if (!response || !response.players) return;

        setResults(response.players.map((p: any) => ({
          id: p.real_id,
          mlb_id: p.mlb_id,
          name: p.name,
          team: p.team,
          position: p.position,
          war_bat: p.war_bat,
          war_pit: p.war_pit,
          is_historical: p.is_historical ?? false,
          career_war: p.career_war,
          first_year: p.first_year,
          last_year: p.last_year,
        })));
      } catch {
        // Silently ignore — transient network errors shouldn't break the search UX
      }
    }, 250);
  };

  const handleSelect = (player: PlayerResult) => {
    setQuery('');
    setResults([]);
    navigate(`/players/${player.mlb_id || player.id}`);
  };

  // Flatten results for keyboard navigation
  const activeResults = results.filter(p => !p.is_historical);
  const historicalResults = results.filter(p => p.is_historical);
  const allFlat = [...activeResults, ...historicalResults];

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (!allFlat.length) return;
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setFocusedIdx(prev => Math.min(prev + 1, allFlat.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setFocusedIdx(prev => Math.max(prev - 1, 0));
    } else if (e.key === 'Enter' && focusedIdx >= 0) {
      e.preventDefault();
      handleSelect(allFlat[focusedIdx]);
    } else if (e.key === 'Escape') {
      setResults([]);
    }
  };

  // Position display: show short position, not full team list
  const formatPosition = (player: PlayerResult) => {
    if (player.is_historical) {
      return player.position === 'P' ? 'P' : 'Pos';
    }
    return player.position || '—';
  };

  let flatIdx = -1;

  return (
    <div className="relative" ref={containerRef}>
      <div className="relative">
        <svg className="absolute left-2.5 top-1/2 -translate-y-1/2 h-3.5 w-3.5 text-surface-500 pointer-events-none" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="2">
          <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
        </svg>
        <input
          type="text"
          value={query}
          onChange={(e) => handleSearch(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Search players..."
          className="w-full pl-8 pr-3 py-1.5 rounded-lg text-xs bg-white/[0.06] border border-white/[0.08] text-surface-200 placeholder-surface-500 
                     focus:outline-none focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40 focus:bg-white/[0.08]
                     transition-all"
        />
      </div>
      {results.length > 0 && (
        <div className="absolute z-50 w-full mt-1 bg-surface-800 border border-white/[0.08] rounded-lg shadow-xl shadow-black/40 overflow-hidden max-h-80 overflow-y-auto">
          {activeResults.map((player) => {
            flatIdx++;
            const idx = flatIdx;
            return (
              <button
                key={`active-${player.id}`}
                onClick={() => handleSelect(player)}
                className={`w-full px-3 py-1.5 text-left flex items-center gap-2 transition-colors border-b border-white/[0.04] last:border-b-0 ${
                  idx === focusedIdx ? 'bg-white/[0.08]' : 'hover:bg-white/[0.06]'
                }`}
              >
                <span className="text-xs font-medium text-white truncate">{player.name}</span>
                <span className="text-[11px] text-surface-500 shrink-0">
                  {player.team?.toUpperCase()} · {formatPosition(player)}
                </span>
              </button>
            );
          })}
          {historicalResults.length > 0 && (
            <>
              {activeResults.length > 0 && (
                <div className="px-3 py-1 text-[10px] text-surface-500 uppercase tracking-wider bg-surface-850 border-y border-white/[0.04]">
                  Historical
                </div>
              )}
              {historicalResults.map((player) => {
                flatIdx++;
                const idx = flatIdx;
                return (
                  <button
                    key={`hist-${player.id}`}
                    onClick={() => handleSelect(player)}
                    className={`w-full px-3 py-1.5 text-left flex items-center gap-2 transition-colors border-b border-white/[0.04] last:border-b-0 ${
                      idx === focusedIdx ? 'bg-white/[0.08]' : 'hover:bg-white/[0.06]'
                    }`}
                  >
                    <span className="text-xs font-medium text-surface-300 truncate">{player.name}</span>
                    <span className="text-[11px] text-surface-500 shrink-0">
                      {formatPosition(player)} · {player.first_year}&ndash;{player.last_year}
                    </span>
                  </button>
                );
              })}
            </>
          )}
        </div>
      )}
    </div>
  );
};

export default PlayerSearch;