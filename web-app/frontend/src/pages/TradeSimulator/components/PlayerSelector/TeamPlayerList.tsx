import React from 'react';
import { Player, Prospect } from '../../../../services/api';

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
  const formatAssetDisplay = (asset: Player | Prospect) => {
    if ('war_bat' in asset || 'war_pit' in asset) {
      const war = asset.war_bat || asset.war_pit || 0;
      return `${asset.name} (${asset.position}) - WAR: ${war.toFixed(1)}`;
    }
    if ('value' in asset && asset.value !== null) {
      return `${asset.name} (${asset.position}) - $${(asset.value / 1000000).toFixed(1)}M`;
    }
    return `${asset.name} (${asset.position || 'Unknown'})`;
  };

  return (
    <div>
      <h3 className="font-bold mb-4 text-white">{team} receives from {otherTeam}:</h3>
      
      <div className="space-y-4">
        {/* Selected Assets */}
        <div className="space-y-2">
          {receivingAssets.map(({ asset, isProspect }) => (
            <div key={asset.name} className="flex justify-between items-center border border-white/[0.06] p-3 rounded-lg bg-white/[0.02]">
              <span className="text-surface-300 text-sm">{formatAssetDisplay(asset)}</span>
              <button 
                onClick={() => onAssetRemove(asset)}
                className="text-red-400 hover:text-red-300 text-sm font-medium"
              >
                Remove
              </button>
            </div>
          ))}
        </div>

        {/* MLB Player Selector */}
        <select
          onChange={(e) => {
            const player = availablePlayers.find(p => p.name === e.target.value);
            if (player) onAssetSelect(player, false);
          }}
          className="w-full bg-surface-700/50 border border-white/[0.08] rounded-lg px-4 py-2.5 text-surface-300 focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40"
          value=""
        >
          <option value="" className="bg-surface-800">Add MLB player...</option>
          {availablePlayers
            .sort((a, b) => ((b.war_bat || b.war_pit || 0) - (a.war_bat || a.war_pit || 0)))
            .map(player => (
              <option key={player.name} value={player.name} className="bg-surface-800">
                {formatAssetDisplay(player)}
              </option>
            ))}
        </select>

        <select
          onChange={(e) => {
            const prospect = availableProspects.find(p => p.name === e.target.value);
            if (prospect) onAssetSelect(prospect, true);
          }}
          className="w-full bg-surface-700/50 border border-white/[0.08] rounded-lg px-4 py-2.5 text-surface-300 focus:ring-2 focus:ring-brand-400/40 focus:border-brand-400/40"
          value=""
        >
          <option value="" className="bg-surface-800">Add prospect...</option>
          {availableProspects
            .sort((a, b) => ((b.value || 0) - (a.value || 0)))
            .map(prospect => (
              <option key={prospect.name} value={prospect.name} className="bg-surface-800">
                {formatAssetDisplay(prospect)}
              </option>
            ))}
        </select>
      </div>
    </div>
  );
};

export default TeamPlayerList;