import React, { useState, useEffect } from 'react'
import { analyzeTrade, getPlayers, Player, TradeAnalysis } from '../../services/api'
import { teamDivisions, sortTeamsByDivision } from '../../config/teams'
import TeamPlayerList from './components/PlayerSelector/TeamPlayerList'
import ValueDisplay from './components/TradeBreakdown/ValueDisplay'

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
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-7xl mx-auto py-16 px-4">
        <h1 className="text-4xl font-bold mb-8 text-gray-900">Trade Analyzer</h1>
        
        {error && (
          <div className="bg-red-50 border-l-4 border-red-400 p-4 mb-6 rounded-r">
            <div className="flex">
              <div className="ml-3">
                <p className="text-red-700">{error}</p>
              </div>
            </div>
          </div>
        )}

        {/* Team Selection */}
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6 mb-8">
          <div className="flex flex-wrap gap-4 items-center">
            {selectedTeams.map((team) => (
              <div key={team} 
                className="flex items-center bg-emerald-50 rounded-lg px-4 py-2 border border-emerald-200"
              >
                <span className="text-emerald-900 font-medium">
                  {team.toUpperCase()} - {teamDivisions[team].name}
                </span>
                <button
                  onClick={() => handleTeamRemove(team)}
                  className="ml-3 text-emerald-600 hover:text-emerald-800"
                >
                  ×
                </button>
              </div>
            ))}
            {selectedTeams.length < 2 && (
              <select
                onChange={(e) => handleTeamAdd(e.target.value)}
                className="border border-gray-300 rounded-lg px-4 py-2 bg-white focus:ring-2 focus:ring-emerald-500 focus:border-emerald-500"
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
        
        {/* Trade Builder */}
        {trade.teamA && trade.teamB && (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <TeamPlayerList
                  team={trade.teamA}
                  availablePlayers={players.filter(p => p.team?.toLowerCase() === trade.teamB?.toLowerCase())}
                  receivingPlayers={trade.teamAReceiving}
                  onPlayerSelect={(player) => handlePlayerAdd(trade.teamA!, player)}
                  onPlayerRemove={(player) => handlePlayerRemove(trade.teamA!, player)}
                  otherTeam={trade.teamB}
                />
              </div>
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <TeamPlayerList
                  team={trade.teamB}
                  availablePlayers={players.filter(p => p.team?.toLowerCase() === trade.teamA?.toLowerCase())}
                  receivingPlayers={trade.teamBReceiving}
                  onPlayerSelect={(player) => handlePlayerAdd(trade.teamB!, player)}
                  onPlayerRemove={(player) => handlePlayerRemove(trade.teamB!, player)}
                  otherTeam={trade.teamA}
                />
              </div>
            </div>
            
            {loading && (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-emerald-500 border-t-transparent"></div>
                <p className="mt-2 text-gray-600">Analyzing trade...</p>
              </div>
            )}
            
            {analysis && (
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <ValueDisplay 
                  analysis={analysis} 
                  team1Name={trade.teamA?.toUpperCase() || ''} 
                  team2Name={trade.teamB?.toUpperCase() || ''}
                />
              </div>
            )}
          </>
        )}
      </div>
    </div>
  )
}

export default TradeAnalyzer