import React, { useState } from 'react'
import { Player, players } from '../../data/mockData'
import TeamSelector from './components/TeamSelector/TeamSelector'
import TeamPlayerList from './components/PlayerSelector/TeamPlayerList'
import ValueDisplay from '../../components/ValueMetrics/display'
import { calculateTeamMetrics } from '../../components/ValueMetrics/calculations'

interface TradeState {
  teamA: string | null;
  teamB: string | null;
  teamAReceiving: Player[];
  teamBReceiving: Player[];
}

const TradeAnalyzer = () => {
  const [trade, setTrade] = useState<TradeState>({
    teamA: null,
    teamB: null,
    teamAReceiving: [],
    teamBReceiving: []
  })

  const handleTeamAdd = (team: string) => {
    if (!trade.teamA) {
      setTrade({ ...trade, teamA: team })
    } else if (!trade.teamB) {
      setTrade({ ...trade, teamB: team })
    }
  }

  const handleTeamRemove = (team: string) => {
    if (trade.teamA === team) {
      setTrade({ 
        ...trade, 
        teamA: null, 
        teamAReceiving: [] 
      })
    } else if (trade.teamB === team) {
      setTrade({ 
        ...trade, 
        teamB: null, 
        teamBReceiving: [] 
      })
    }
  }

  const handlePlayerAdd = (receivingTeam: string, player: Player) => {
    if (receivingTeam === trade.teamA) {
      setTrade({
        ...trade,
        teamAReceiving: [...trade.teamAReceiving, player]
      })
    } else {
      setTrade({
        ...trade,
        teamBReceiving: [...trade.teamBReceiving, player]
      })
    }
  }

  const handlePlayerRemove = (receivingTeam: string, player: Player) => {
    if (receivingTeam === trade.teamA) {
      setTrade({
        ...trade,
        teamAReceiving: trade.teamAReceiving.filter(p => p.id !== player.id)
      })
    } else {
      setTrade({
        ...trade,
        teamBReceiving: trade.teamBReceiving.filter(p => p.id !== player.id)
      })
    }
  }

  const selectedTeams = [trade.teamA, trade.teamB].filter(Boolean) as string[]

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">MLB Trade Analyzer</h1>
      <TeamSelector
        selectedTeams={selectedTeams}
        onTeamAdd={handleTeamAdd}
        onTeamRemove={handleTeamRemove}
        maxTeams={2}
      />
      {trade.teamA && trade.teamB && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <TeamPlayerList
              team={trade.teamA}
              availablePlayers={players.filter(p => p.team === trade.teamB)}
              receivingPlayers={trade.teamAReceiving}
              onPlayerSelect={(player) => handlePlayerAdd(trade.teamA!, player)}
              onPlayerRemove={(player) => handlePlayerRemove(trade.teamA!, player)}
              otherTeam={trade.teamB}
            />
            <TeamPlayerList
              team={trade.teamB}
              availablePlayers={players.filter(p => p.team === trade.teamA)}
              receivingPlayers={trade.teamBReceiving}
              onPlayerSelect={(player) => handlePlayerAdd(trade.teamB!, player)}
              onPlayerRemove={(player) => handlePlayerRemove(trade.teamB!, player)}
              otherTeam={trade.teamA}
            />
          </div>
          <ValueDisplay
            metrics={[
              calculateTeamMetrics(trade.teamA, trade.teamAReceiving),
              calculateTeamMetrics(trade.teamB, trade.teamBReceiving)
            ]}
          />
        </>
      )}
    </div>
  )
}

export default TradeAnalyzer