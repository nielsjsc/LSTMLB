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
            <div key={asset.name} className="flex justify-between items-center border border-slate-700 p-2 rounded">
              <span className="text-gray-300">{formatAssetDisplay(asset)}</span>
              <button 
                onClick={() => onAssetRemove(asset)}
                className="text-red-400 hover:text-red-300"
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
          className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 text-gray-300"
          value=""
        >
          <option value="" className="bg-slate-800">Add MLB player...</option>
          {availablePlayers
            .sort((a, b) => ((b.war_bat || b.war_pit || 0) - (a.war_bat || a.war_pit || 0)))
            .map(player => (
              <option key={player.name} value={player.name} className="bg-slate-800">
                {formatAssetDisplay(player)}
              </option>
            ))}
        </select>

        <select
          onChange={(e) => {
            const prospect = availableProspects.find(p => p.name === e.target.value);
            if (prospect) onAssetSelect(prospect, true);
          }}
          className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 text-gray-300"
          value=""
        >
          <option value="" className="bg-slate-800">Add prospect...</option>
          {availableProspects
            .sort((a, b) => ((b.value || 0) - (a.value || 0)))
            .map(prospect => (
              <option key={prospect.name} value={prospect.name} className="bg-slate-800">
                {formatAssetDisplay(prospect)}
              </option>
            ))}
        </select>
      </div>
    </div>
  );
};

export default TeamPlayerList;