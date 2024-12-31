import React, { useState, useEffect } from 'react'
import { analyzeTrade, getPlayers, Player, TradeAnalysis } from '../../services/api'
import { teamDivisions, sortTeamsByDivision } from '../../config/teams'
import TeamPlayerList from './components/PlayerSelector/TeamPlayerList'
import ValueDisplay from '../../components/ValueMetrics/display'

interface TradeState {
  teamA: string | null;
  teamB: string | null;
  teamAReceiving: Player[];
  teamBReceiving: Player[];
}

const TradeAnalyzer = () => {
  const [players, setPlayers] = useState<Player[]>([])
  const [trade, setTrade] = useState<TradeState>({
    teamA: null,
    teamB: null,
    teamAReceiving: [],
    teamBReceiving: []
  })
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [analysis, setAnalysis] = useState<TradeAnalysis | null>(null)

  useEffect(() => {
    const fetchPlayers = async () => {
      try {
        const data = await getPlayers(2025);
        setPlayers(data);
      } catch (err) {
        setError('Failed to load players');
      }
    };
    fetchPlayers();
  }, []);

  const handleTeamAdd = (team: string) => {
    if (!trade.teamA) {
      setTrade({ ...trade, teamA: team });
    } else if (!trade.teamB) {
      setTrade({ ...trade, teamB: team });
    }
  }

  const handleTeamRemove = (team: string) => {
    if (trade.teamA === team) {
      setTrade({ ...trade, teamA: null, teamAReceiving: [] });
    } else if (trade.teamB === team) {
      setTrade({ ...trade, teamB: null, teamBReceiving: [] });
    }
    setAnalysis(null);
  }

  const handlePlayerAdd = async (receivingTeam: string, player: Player) => {
    let updatedTrade: TradeState;
    if (receivingTeam === trade.teamA) {
      updatedTrade = {
        ...trade,
        teamAReceiving: [...trade.teamAReceiving, player]
      };
    } else {
      updatedTrade = {
        ...trade,
        teamBReceiving: [...trade.teamBReceiving, player]
      };
    }
    setTrade(updatedTrade);
    
    if (updatedTrade.teamAReceiving.length > 0 || updatedTrade.teamBReceiving.length > 0) {
      await handleAnalyzeTrade(updatedTrade);
    }
  }

  const handlePlayerRemove = async (receivingTeam: string, player: Player) => {
    let updatedTrade: TradeState;
    if (receivingTeam === trade.teamA) {
      updatedTrade = {
        ...trade,
        teamAReceiving: trade.teamAReceiving.filter(p => p.name !== player.name)
      };
    } else {
      updatedTrade = {
        ...trade,
        teamBReceiving: trade.teamBReceiving.filter(p => p.name !== player.name)
      };
    }
    setTrade(updatedTrade);
    
    if (updatedTrade.teamAReceiving.length > 0 || updatedTrade.teamBReceiving.length > 0) {
      await handleAnalyzeTrade(updatedTrade);
    }
  }

  const handleAnalyzeTrade = async (currentTrade: TradeState = trade) => {
    setLoading(true);
    try {
      const result = await analyzeTrade(
        currentTrade.teamAReceiving.map(p => p.name),
        currentTrade.teamBReceiving.map(p => p.name)
      );
      setAnalysis(result);
      setError(null);
    } catch (err) {
      setError('Failed to analyze trade');
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  }

  const selectedTeams = [trade.teamA, trade.teamB].filter(Boolean) as string[];

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">MLB Trade Analyzer</h1>
      
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          {error}
        </div>
      )}

      <div className="mb-6">
        <div className="flex flex-wrap gap-4">
          {selectedTeams.map((team) => (
            <div key={team} className="flex items-center bg-gray-100 rounded-lg p-2">
              <span className="mr-2">{team.toUpperCase()} - {teamDivisions[team].name}</span>
              <button
                onClick={() => handleTeamRemove(team)}
                className="text-red-500 hover:text-red-700"
              >
                ×
              </button>
            </div>
          ))}
          {selectedTeams.length < 2 && (
            <select
              onChange={(e) => handleTeamAdd(e.target.value)}
              className="border rounded-lg p-2"
              value=""
            >
              <option value="">Select team...</option>
              {sortTeamsByDivision(Object.keys(teamDivisions)
                .filter(team => !selectedTeams.includes(team)))
                .map((team) => (
                  <option key={team} value={team}>
                    {team.toUpperCase()} - {teamDivisions[team].name}
                  </option>
              ))}
            </select>
          )}
        </div>
      </div>
      
      {trade.teamA && trade.teamB && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            <TeamPlayerList
              team={trade.teamA}
              availablePlayers={players.filter(p => p.team?.toLowerCase() === trade.teamB?.toLowerCase())}
              receivingPlayers={trade.teamAReceiving}
              onPlayerSelect={(player) => handlePlayerAdd(trade.teamA!, player)}
              onPlayerRemove={(player) => handlePlayerRemove(trade.teamA!, player)}
              otherTeam={trade.teamB}
            />
            <TeamPlayerList
              team={trade.teamB}
              availablePlayers={players.filter(p => p.team?.toLowerCase() === trade.teamA?.toLowerCase())}
              receivingPlayers={trade.teamBReceiving}
              onPlayerSelect={(player) => handlePlayerAdd(trade.teamB!, player)}
              onPlayerRemove={(player) => handlePlayerRemove(trade.teamB!, player)}
              otherTeam={trade.teamA}
            />
          </div>
          
          {loading && <div className="text-center py-4">Analyzing trade...</div>}
          
          {analysis && <ValueDisplay analysis={analysis} />}
        </>
      )}
    </div>
  )
}

export default TradeAnalyzer