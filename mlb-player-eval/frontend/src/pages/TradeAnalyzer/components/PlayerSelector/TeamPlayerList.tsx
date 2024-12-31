import React from 'react'
import { Player } from '../../../../services/api'

interface TeamPlayerListProps {
  team: string;
  availablePlayers: Player[];
  receivingPlayers: Player[];
  onPlayerSelect: (player: Player) => void;
  onPlayerRemove: (player: Player) => void;
  otherTeam: string;
}

const TeamPlayerList: React.FC<TeamPlayerListProps> = ({
  team,
  availablePlayers,
  receivingPlayers,
  onPlayerSelect,
  onPlayerRemove,
  otherTeam
}) => {
  // Filter out already selected players
  const unselectedPlayers = availablePlayers.filter(player => 
    !receivingPlayers.some(selected => selected.name === player.name)
  );

  // Sort by WAR
  const sortedPlayers = [...unselectedPlayers].sort((a, b) => b.war - a.war);

  const formatWar = (war: number) => war.toFixed(1);

  return (
    <div className="border rounded-lg p-4">
      <h3 className="font-bold mb-4">{team} receives from {otherTeam}:</h3>
      
      <div className="space-y-2">
        {receivingPlayers.map(player => (
          <div key={player.name} className="flex justify-between items-center bg-gray-50 p-2 rounded">
            <div>
              <span className="font-semibold">{player.name}</span>
              <div className="text-sm text-gray-600">
                WAR: {formatWar(player.war)}
              </div>
            </div>
            <button
              onClick={() => onPlayerRemove(player)}
              className="text-red-500 hover:text-red-700"
            >
              Remove
            </button>
          </div>
        ))}

        <select
          onChange={(e) => {
            const player = sortedPlayers.find(p => p.name === e.target.value);
            if (player) onPlayerSelect(player);
          }}
          className="w-full border rounded p-2 mt-4"
          value=""
        >
          <option value="">Add player...</option>
          {sortedPlayers.map(player => (
            <option key={player.name} value={player.name}>
              {player.name} (WAR: {formatWar(player.war)})
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};

export default TeamPlayerList;