import React from 'react'
import { Player } from '../../../../data/mockData'

interface TeamPlayerListProps {
  team: string;
  players: Player[];
  selectedPlayers: Player[];
  onPlayerSelect: (player: Player) => void;
  onPlayerRemove: (player: Player) => void;
}

const TeamPlayerList = ({ 
  team, 
  players, 
  selectedPlayers, 
  onPlayerSelect, 
  onPlayerRemove 
}: TeamPlayerListProps) => {
  const teamPlayers = players.filter(p => p.team === team)

  return (
    <div className="border rounded-lg p-4">
      <h3 className="font-bold mb-4">{team}</h3>
      <div className="space-y-2">
        {selectedPlayers.map(player => (
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
            const player = teamPlayers.find(p => p.id === e.target.value)
            if (player) onPlayerSelect(player)
          }}
          className="w-full border rounded p-2"
          value=""
        >
          <option value="">Add player...</option>
          {teamPlayers
            .filter(p => !selectedPlayers.some(sp => sp.id === p.id))
            .map(player => (
              <option key={player.id} value={player.id}>
                {player.name}
              </option>
            ))
          }
        </select>
      </div>
    </div>
  )
}

export default TeamPlayerList