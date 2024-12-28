import { useState, useMemo } from 'react'
import { Player, players } from '../data/mockData'

export const usePlayerData = () => {
  const [search, setSearch] = useState('')
  const [position, setPosition] = useState('All')
  const [team, setTeam] = useState('All')

  const filteredPlayers = useMemo(() => {
    return players.filter(player => {
      const matchesSearch = player.name.toLowerCase().includes(search.toLowerCase())
      const matchesPosition = position === 'All' || player.position === position
      const matchesTeam = team === 'All' || player.team === team
      return matchesSearch && matchesPosition && matchesTeam
    })
  }, [search, position, team])

  return {
    players: filteredPlayers,
    search,
    setSearch,
    position,
    setPosition,
    team,
    setTeam
  }
}