import React from 'react'
import { Player } from '../../data/mockData'

interface PlayerCardProps {
  player: Player;
}

const PlayerCard = ({ player }: PlayerCardProps) => {
  return (
    <div className="border rounded-lg p-4 shadow-sm">
      <h3 className="font-bold text-lg">{player.name}</h3>
      <p>{player.team} | {player.position}</p>
      <div className="mt-2">
        <p>Value: ${player.value}M</p>
        <p>WAR: {player.war}</p>
        <p>Salary: ${player.salary}M</p>
      </div>
    </div>
  )
}

export default PlayerCard