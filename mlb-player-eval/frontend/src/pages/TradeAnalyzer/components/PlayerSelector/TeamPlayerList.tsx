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
  const unselectedPlayers = availablePlayers.filter(player => 
    !receivingPlayers.some(selected => selected.name === player.name)
  );

  const sortedPlayers = [...unselectedPlayers].sort((a, b) => {
    const warA = a.war_bat ?? a.war_pit ?? 0;
    const warB = b.war_bat ?? b.war_pit ?? 0;
    return warB - warA;
  });

  const formatWar = (player: Player): string => {
    const war = player.war_bat ?? player.war_pit;
    return war != null ? war.toFixed(1) : '-';
  };

  return (
    <div className="border rounded-lg p-4">
      <h3 className="font-bold mb-4">{team} receives from {otherTeam}:</h3>
      
      <div className="space-y-2">
        {receivingPlayers.map(player => (
          <div key={player.id} className="flex justify-between items-center bg-gray-50 p-2 rounded">
            <div>
              <span className="font-semibold">{player.name}</span>
              <div className="text-sm text-gray-600">
                WAR: {formatWar(player)}
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
            const player = sortedPlayers.find(p => p.id === parseInt(e.target.value));
            if (player) onPlayerSelect(player);
          }}
          className="w-full border rounded p-2 mt-4"
          value=""
        >
          <option value="">Add player...</option>
          {sortedPlayers.map(player => (
            <option key={player.id} value={player.id}>
              {player.name} (WAR: {formatWar(player)})
            </option>
          ))}
        </select>
      </div>
    </div>
  );
};

export default TeamPlayerList;