import React, { useState } from 'react'
import { Player, players } from '../../data/mockData'
import TeamSelector from './components/TeamSelector/TeamSelector'
import TeamPlayerList from './components/PlayerSelector/TeamPlayerList'
import ValueDisplay from './components/TradeBreakdown/ValueDisplay'


interface TeamTrade {
  team: string;
  players: Player[];
  totalValue: number;
}

const TradeAnalyzer = () => {
  const [selectedTeams, setSelectedTeams] = useState<string[]>([])
  const [teamPlayers, setTeamPlayers] = useState<{ [key: string]: Player[] }>({})

  const handleTeamAdd = (team: string) => {
    setSelectedTeams([...selectedTeams, team])
    setTeamPlayers({ ...teamPlayers, [team]: [] })
  }

  const handleTeamRemove = (team: string) => {
    setSelectedTeams(selectedTeams.filter(t => t !== team))
    const { [team]: removed, ...rest } = teamPlayers
    setTeamPlayers(rest)
  }

  const handlePlayerSelect = (team: string, player: Player) => {
    setTeamPlayers({
      ...teamPlayers,
      [team]: [...(teamPlayers[team] || []), player]
    })
  }

  const handlePlayerRemove = (team: string, player: Player) => {
    setTeamPlayers({
      ...teamPlayers,
      [team]: teamPlayers[team].filter(p => p.id !== player.id)
    })
  }

  const tradeValues: TeamTrade[] = selectedTeams.map(team => ({
    team,
    players: teamPlayers[team] || [],
    totalValue: (teamPlayers[team] || []).reduce((sum, p) => sum + p.value, 0)
  }))

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">MLB Trade Analyzer</h1>
      <TeamSelector
        selectedTeams={selectedTeams}
        onTeamAdd={handleTeamAdd}
        onTeamRemove={handleTeamRemove}
      />
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-6">
        {selectedTeams.map(team => (
          <TeamPlayerList
            key={team}
            team={team}
            players={players}
            selectedPlayers={teamPlayers[team] || []}
            onPlayerSelect={(player) => handlePlayerSelect(team, player)}
            onPlayerRemove={(player) => handlePlayerRemove(team, player)}
          />
        ))}
      </div>
      {selectedTeams.length > 0 && <ValueDisplay teams={tradeValues} />}
    </div>
  )
}

export default TradeAnalyzer