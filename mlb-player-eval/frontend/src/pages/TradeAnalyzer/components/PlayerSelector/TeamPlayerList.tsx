import React from 'react'
import { Player } from '../../../../data/mockData'

interface TeamPlayerListProps {
  team: string;
  availablePlayers: Player[];
  receivingPlayers: Player[];
  onPlayerSelect: (player: Player) => void;
  onPlayerRemove: (player: Player) => void;
  otherTeam: string;
}

const TeamPlayerList = ({ 
  team, 
  availablePlayers, 
  receivingPlayers, 
  onPlayerSelect, 
  onPlayerRemove,
  otherTeam 
}: TeamPlayerListProps) => {
  return (
    <div className="border rounded-lg p-4">
      <h3 className="font-bold mb-4">{team}</h3>
      <p className="text-sm mb-2">Receiving from {otherTeam}:</p>
      <div className="space-y-2">
        {receivingPlayers.map(player => (
          <div key={player.id} className="flex justify-between items-center bg-gray-50 p-2 rounded">
            <span>{player.name}</span>
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
            const player = availablePlayers.find(p => p.id === e.target.value)
            if (player) onPlayerSelect(player)
          }}
          className="w-full border rounded p-2"
          value=""
        >
          <option value="">Add player...</option>
          {availablePlayers
            .sort((a, b) => b.war - a.war)
            .map(player => (
              <option key={player.id} value={player.id}>
                {player.name} (WAR: {player.war})
              </option>
            ))
          }
        </select>
      </div>
    </div>
  )
}

export default TeamPlayerList