import React from 'react'
import { usePlayerData } from '../../hooks/usePlayerData'
import SearchBar from '../../components/Search/SearchBar'
import PlayerGrid from '../../components/Player/PlayerGrid'
import FilterContainer from '../../components/Player/Filters/FilterContainer'

const Players = () => {
  const { 
    players, 
    search, 
    setSearch, 
    position, 
    setPosition, 
    team, 
    setTeam 
  } = usePlayerData()

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">MLB Players</h1>
      <SearchBar value={search} onChange={setSearch} />
      <FilterContainer
        selectedPosition={position}
        selectedTeam={team}
        onPositionChange={setPosition}
        onTeamChange={setTeam}
      />
      <PlayerGrid players={players} />
    </div>
  )
}

export default Players