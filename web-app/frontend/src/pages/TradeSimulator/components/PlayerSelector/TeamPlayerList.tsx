import React, { useState, useRef } from 'react';
import { Link } from 'react-router-dom';
import { Player, Prospect } from '../../../../services/api';
import { getTeamColors } from '../../../../utils/teamColors';

interface TeamPlayerListProps {
  team: string;
  availablePlayers: Player[];
  availableProspects: Prospect[];
  receivingAssets: Array<{ asset: Player | Prospect; isProspect: boolean }>;
  onAssetSelect: (asset: Player | Prospect, isProspect: boolean) => void;
  onAssetRemove: (asset: Player | Prospect) => void;
  otherTeam: string;
}

const TeamPlayerList: React.FC<TeamPlayerListProps> = ({
  team,
  availablePlayers,
  availableProspects,
  receivingAssets,
  onAssetSelect,
  onAssetRemove,
  otherTeam
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'mlb' | 'prospects'>('mlb');
  const inputRef = useRef<HTMLInputElement>(null);

  const otherTeamColors = getTeamColors(otherTeam);

  const getPositionColor = (position: string | undefined) => {
    if (!position) return 'bg-gray-200 text-gray-600';
    const pos = position.toUpperCase();
    if (['SP', 'RP', 'CL', 'P'].includes(pos)) return 'bg-blue-500/20 text-blue-400 border border-blue-500/20';
    if (['C', '1B', '2B', '3B', 'SS'].includes(pos)) return 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/20';
    if (['LF', 'CF', 'RF', 'OF'].includes(pos)) return 'bg-amber-500/20 text-amber-400 border border-amber-500/20';
    if (['DH'].includes(pos)) return 'bg-purple-500/20 text-purple-400 border border-purple-500/20';
    return 'bg-gray-200/60 text-gray-600 border border-gray-300';
  };

  const filteredPlayers = availablePlayers
    .filter(p => !receivingAssets.some(a => a.asset.name === p.name))
    .filter(p => p.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => ((b.war_bat || b.war_pit || 0) - (a.war_bat || a.war_pit || 0)));

  const filteredProspects = availableProspects
    .filter(p => !receivingAssets.some(a => a.asset.name === p.name))
    .filter(p => p.name.toLowerCase().includes(searchQuery.toLowerCase()))
    .sort((a, b) => ((b.value || 0) - (a.value || 0)));

  const totalValue = receivingAssets.reduce((sum, { asset }) => {
    if ('war_bat' in asset || 'war_pit' in asset) {
      const baseValue = (asset as Player).base_value;
      return sum + (baseValue && !isNaN(baseValue) ? baseValue : 0);
    }
    const value = (asset as Prospect).value;
    return sum + (value && !isNaN(value) ? value : 0);
  }, 0);

  return (
    <div className="flex flex-col h-full">
      {/* Header with team branding */}
      <div className="flex items-center gap-3 mb-4">
        <div className="w-1 h-8 rounded-full" style={{ background: otherTeamColors.primary }} />
        <div>
          <h3 className="font-semibold text-gray-900 text-sm tracking-wide">
            {team.toUpperCase()} receives from {otherTeam.toUpperCase()}
          </h3>
          <p className="text-xs text-gray-500">
            {receivingAssets.length} player{receivingAssets.length !== 1 ? 's' : ''} added
          </p>
        </div>
      </div>

      {/* Selected Assets as Cards */}
      <div className="space-y-2 mb-4 min-h-[60px]">
        {receivingAssets.length === 0 ? (
          <div className="flex items-center justify-center h-[60px] rounded-lg border border-dashed border-gray-200 bg-white/[0.01]">
            <p className="text-gray-400 text-xs">Search below to add players</p>
          </div>
        ) : (
          receivingAssets.map(({ asset, isProspect }) => {
            const war = 'war_bat' in asset ? asset.war_bat : 'war_pit' in asset ? asset.war_pit : null;
            const value = 'value' in asset ? (asset as Prospect).value : null;
            const baseValue = 'base_value' in asset ? (asset as Player).base_value : null;

            return (
              <div
                key={'playerId' in asset ? asset.playerId : ('mlb_id' in asset ? asset.mlb_id : asset.real_id)}
                className="group relative flex items-center gap-3 p-3 rounded-lg border border-gray-200 bg-white/[0.02] hover:bg-gray-50 transition-all duration-200"
              >
                {/* Position badge */}
                <span className={`shrink-0 px-2 py-0.5 rounded text-[10px] font-bold uppercase ${getPositionColor(asset.position)}`}>
                  {asset.position || '?'}
                </span>

                {/* Player info */}
                <div className="flex-1 min-w-0">
                  {'id' in asset ? (
                    <Link 
                      to={`/players/${(asset as Player).id}`}
                      className="text-sm font-medium text-gray-900 hover:text-brand-500 transition-colors truncate block"
                    >
                      {asset.name}
                    </Link>
                  ) : (
                    <p className="text-sm font-medium text-gray-900 truncate">{asset.name}</p>
                  )}
                  <div className="flex items-center gap-2 mt-0.5">
                    {isProspect ? (
                      <>
                        <span className="text-[10px] font-medium text-purple-400 bg-purple-500/10 px-1.5 py-0.5 rounded">
                          PROSPECT
                        </span>
                        {('fv' in asset) && (
                          <span className="text-xs text-gray-500">FV: {(asset as Prospect).fv}</span>
                        )}
                      </>
                    ) : (
                      <>
                        {war !== null && war !== undefined && (
                          <span className="text-xs text-gray-500">
                            {(war as number).toFixed(1)} WAR
                          </span>
                        )}
                      </>
                    )}
                    {value !== null && value !== undefined && !isNaN(value as number) && (
                      <span className={`text-xs font-medium ${(value as number) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        ${((value as number) / 1_000_000).toFixed(1)}M
                      </span>
                    )}
                    {baseValue !== null && baseValue !== undefined && !isNaN(baseValue as number) && (
                      <span className={`text-xs font-medium ${(baseValue as number) >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
                        ${((baseValue as number) / 1_000_000).toFixed(1)}M
                      </span>
                    )}
                  </div>
                </div>

                {/* Remove button */}
                <button
                  onClick={() => onAssetRemove(asset)}
                  className="shrink-0 w-6 h-6 flex items-center justify-center rounded-full text-gray-400 hover:text-red-400 hover:bg-red-500/10 transition-all duration-200 opacity-0 group-hover:opacity-100"
                  title="Remove player"
                >
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            );
          })
        )}
      </div>

      {/* Total value bar */}
      {receivingAssets.length > 0 && !isNaN(totalValue) && (
        <div className="flex items-center justify-between px-3 py-2 rounded-lg bg-white/[0.02] border border-gray-100 mb-4">
          <span className="text-xs text-gray-500 font-medium">Package Value</span>
          <span className={`text-sm font-bold ${totalValue >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            ${(totalValue / 1_000_000).toFixed(1)}M
          </span>
        </div>
      )}

      {/* Player List */}
      <div className="flex flex-col">
        {/* Tab switcher */}
        <div className="flex mb-2 bg-white/80 rounded-lg p-0.5 border border-gray-100">
          <button
            onClick={() => { setActiveTab('mlb'); setSearchQuery(''); }}
            className={`flex-1 text-xs font-medium py-1.5 rounded-md transition-all duration-200 ${
              activeTab === 'mlb'
                ? 'bg-gray-100 text-gray-900 shadow-sm'
                : 'text-gray-500 hover:text-gray-600'
            }`}
          >
            MLB Players ({filteredPlayers.length})
          </button>
          <button
            onClick={() => { setActiveTab('prospects'); setSearchQuery(''); }}
            className={`flex-1 text-xs font-medium py-1.5 rounded-md transition-all duration-200 ${
              activeTab === 'prospects'
                ? 'bg-gray-100 text-gray-900 shadow-sm'
                : 'text-gray-500 hover:text-gray-600'
            }`}
          >
            Prospects ({filteredProspects.length})
          </button>
        </div>

        {/* Search input */}
        <div className="relative mb-2">
          <svg className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
          </svg>
          <input
            ref={inputRef}
            type="text"
            placeholder={activeTab === 'mlb' ? 'Search MLB players...' : 'Search prospects...'}
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full bg-white border border-gray-200 rounded-lg pl-9 pr-4 py-2.5 text-sm text-gray-600 
              placeholder:text-gray-400 focus:ring-2 focus:ring-brand-500/30 focus:border-brand-200 transition-all duration-200"
          />
        </div>

        {/* Player list - always visible */}
        <div className="bg-white/95 backdrop-blur-sm border border-white/[0.15] rounded-lg shadow-lg max-h-[320px] overflow-y-auto">
            {activeTab === 'mlb' ? (
              filteredPlayers.length === 0 ? (
                <div className="px-4 py-6 text-center text-gray-400 text-xs">No players found</div>
              ) : (
                filteredPlayers.slice(0, 50).map(player => {
                  const war = player.war_bat || player.war_pit || 0;
                  return (
                    <button
                      key={player.mlb_id || player.real_id}
                      onClick={() => {
                        onAssetSelect(player, false);
                        setSearchQuery('');
                      }}
                      className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-brand-400/10 transition-colors duration-100 border-b border-gray-200 last:border-0"
                    >
                      <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${getPositionColor(player.position)}`}>
                        {player.position}
                      </span>
                      <span className="flex-1 text-sm text-gray-800 truncate">{player.name}</span>
                      <span className={`text-xs font-medium tabular-nums ${war >= 3 ? 'text-emerald-400' : war >= 1 ? 'text-gray-600' : 'text-gray-500'}`}>
                        {war.toFixed(1)} WAR
                      </span>
                    </button>
                  );
                })
              )
            ) : (
              filteredProspects.length === 0 ? (
                <div className="px-4 py-6 text-center text-gray-400 text-xs">No prospects found</div>
              ) : (
                filteredProspects.slice(0, 50).map(prospect => (
                  <button
                    key={prospect.playerId}
                    onClick={() => {
                      onAssetSelect(prospect, true);
                      setSearchQuery('');
                    }}
                    className="w-full flex items-center gap-3 px-3 py-2.5 text-left hover:bg-brand-400/10 transition-colors duration-100 border-b border-gray-200 last:border-0"
                  >
                    <span className={`shrink-0 px-1.5 py-0.5 rounded text-[10px] font-bold uppercase ${getPositionColor(prospect.position)}`}>
                      {prospect.position}
                    </span>
                    <span className="flex-1 text-sm text-gray-800 truncate">{prospect.name}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-[10px] text-purple-400 bg-purple-500/10 px-1 py-0.5 rounded font-medium">
                        FV {prospect.fv}
                      </span>
                      {prospect.value && !isNaN(prospect.value) && (
                        <span className="text-xs font-medium text-gray-500 tabular-nums">
                          ${(prospect.value / 1_000_000).toFixed(1)}M
                        </span>
                      )}
                    </div>
                  </button>
                ))
              )
            )}
          </div>
      </div>
    </div>
  );
};

export default TeamPlayerList;