import React from 'react'
import { Player } from '../../../../data/mockData'

interface TeamValue {
  team: string;
  players: Player[];
  totalValue: number;
}

interface ValueDisplayProps {
  teams: TeamValue[];
}

const ValueDisplay = ({ teams }: ValueDisplayProps) => {
  return (
    <div className="mt-6 border-t pt-6">
      <h2 className="text-xl font-semibold mb-4">Trade Analysis</h2>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {teams.map(({ team, players, totalValue }) => (
          <div key={team} className="border rounded-lg p-4">
            <h3 className="font-bold mb-2">{team}</h3>
            <p className="text-lg font-semibold text-green-600">
              Total Value: ${totalValue.toFixed(1)}M
            </p>
            <div className="mt-2">
              {players.map(player => (
                <div key={player.id} className="text-sm text-gray-600">
                  {player.name}: ${player.value}M
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}

export default ValueDisplay