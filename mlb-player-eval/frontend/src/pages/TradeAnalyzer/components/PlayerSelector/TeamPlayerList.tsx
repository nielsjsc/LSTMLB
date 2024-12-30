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
  console.log('TeamPlayerList Input:', {
      team,
      otherTeam,
      allPlayers: availablePlayers,
      allPlayersLength: availablePlayers?.length,
      somePlayerTeams: availablePlayers?.slice(0,5).map(p => p.team)
  });

  // Remove filter initially to see if we get any players
  const sortedPlayers = [...availablePlayers].sort((a, b) => b.war - a.war);

  return (
      <div className="border rounded-lg p-4">
          <h3 className="font-bold mb-4">{team} receives from {otherTeam}:</h3>
          <div className="space-y-2">
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
                          {player.name} ({player.team}) - WAR: {player.war}
                      </option>
                  ))}
              </select>
          </div>
      </div>
  );
};

export default TeamPlayerList;