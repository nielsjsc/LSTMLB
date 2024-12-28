import React from 'react'
import { Player } from '../../data/mockData'
import PlayerCard from './PlayerCard'

interface PlayerGridProps {
  players: Player[];
}

const PlayerGrid = ({ players }: PlayerGridProps) => {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
      {players.map((player) => (
        <PlayerCard key={player.id} player={player} />
      ))}
    </div>
  )
}

export default PlayerGrid