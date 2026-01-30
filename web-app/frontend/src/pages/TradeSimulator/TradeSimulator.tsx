import React, { useState, useEffect } from 'react';
import { 
  analyzeTrade, 
  getPlayers, 
  getAllProspects,
  Player, 
  Prospect,
  TradeAnalysis 
} from '../../services/api';
import { teamDivisions, sortTeamsByDivision } from '../../config/teams';
import TeamPlayerList from './components/PlayerSelector/TeamPlayerList';
import ValueDisplay from './components/TradeBreakdown/ValueDisplay';

interface TradeState {
  teamA: string | null;
  teamB: string | null;
  teamAReceiving: Array<{ asset: Player | Prospect; isProspect: boolean }>;
  teamBReceiving: Array<{ asset: Player | Prospect; isProspect: boolean }>;
}

const TradeAnalyzer = () => {
  const [players, setPlayers] = useState<Player[]>([]);
  const [prospects, setProspects] = useState<Prospect[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<TradeAnalysis | null>(null);
  const [trade, setTrade] = useState<TradeState>({
    teamA: null,
    teamB: null,
    teamAReceiving: [],
    teamBReceiving: []
  });

  useEffect(() => {
    const fetchAssets = async () => {
      try {
        const [playersData, hittersData, pitchersData] = await Promise.all([
          getPlayers(CURRENT_YEAR),
          getAllProspects('hitter', 2025),
          getAllProspects('pitcher', 2025)
        ]);
        
        setPlayers(playersData);
        const allProspects = [...hittersData, ...pitchersData]
          .sort((a, b) => (b.value || 0) - (a.value || 0));
        
        console.log(`Loaded ${playersData.length} players and ${allProspects.length} prospects`);
        setProspects(allProspects);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Unknown error';
        console.error('Error fetching assets:', errorMessage);
        setError(`Failed to load assets: ${errorMessage}`);
      }
    };
  
    fetchAssets();
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

  const handleAssetAdd = async (receivingTeam: string, asset: Player | Prospect, isProspect: boolean) => {
    let updatedTrade: TradeState;
    if (receivingTeam === trade.teamA) {
      updatedTrade = {
        ...trade,
        teamAReceiving: [...trade.teamAReceiving, { asset, isProspect }]
      };
    } else {
      updatedTrade = {
        ...trade,
        teamBReceiving: [...trade.teamBReceiving, { asset, isProspect }]
      };
    }
    setTrade(updatedTrade);
    await handleAnalyzeTrade(updatedTrade);
  };

  // Update handler to work with both types
  const handleAssetRemove = async (receivingTeam: string, asset: Player | Prospect) => {
    let updatedTrade: TradeState;
    if (receivingTeam === trade.teamA) {
      updatedTrade = {
        ...trade,
        teamAReceiving: trade.teamAReceiving.filter(a => a.asset.name !== asset.name)
      };
    } else {
      updatedTrade = {
        ...trade,
        teamBReceiving: trade.teamBReceiving.filter(a => a.asset.name !== asset.name)
      };
    }
    setTrade(updatedTrade);
    await handleAnalyzeTrade(updatedTrade);
  };

  const handleAnalyzeTrade = async (currentTrade: TradeState = trade) => {
    if (!currentTrade.teamA || !currentTrade.teamB) return;
    
    setLoading(true);
    try {
      const team1Assets = currentTrade.teamAReceiving.map(a => ({
        name: a.asset.name,
        isProspect: a.isProspect,
        team: currentTrade.teamA!.toLowerCase()
      }));
  
      const team2Assets = currentTrade.teamBReceiving.map(a => ({
        name: a.asset.name,
        isProspect: a.isProspect,
        team: currentTrade.teamB!.toLowerCase()
      }));
  
      const result = await analyzeTrade(team1Assets, team2Assets);
      setAnalysis(result);
      setError(null);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to analyze trade';
      console.error('Trade analysis error:', errorMessage);
      setError(errorMessage);
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  };
  const selectedTeams = [trade.teamA, trade.teamB].filter(Boolean) as string[];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto py-16 px-4">
        {/* Header */}
        <div className="text-center mb-16">
          <h1 className="text-4xl font-bold mb-4 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-emerald-400 pb-2">
            Trade Simulator
          </h1>
          <p className="text-gray-400 max-w-2xl mx-auto">
            Evaluate trades using our projection-based player valuations and prospect rankings
          </p>
        </div>

        {error && (
          <div className="border border-red-500/20 rounded-lg p-4 mb-6">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {/* Team Selection */}
        <div className="grid md:grid-cols-2 gap-8 mb-12">
          {/* Team A */}
          <div className="rounded-xl p-6 border border-slate-700/50 bg-slate-800/50">
            <h3 className="text-lg font-medium text-white mb-4">Select Team 1</h3>
            {trade.teamA ? (
              <div className="flex items-center justify-between rounded-lg px-4 py-3 border border-emerald-500/20 bg-slate-700/50">
                <span className="text-emerald-400 font-medium">
                  {trade.teamA.toUpperCase()} - {teamDivisions[trade.teamA].name}
                </span>
                <button
                  onClick={() => handleTeamRemove(trade.teamA!)}
                  className="text-emerald-400 hover:text-emerald-300"
                >
                  ×
                </button>
              </div>
            ) : (
              <select
                onChange={(e) => handleTeamAdd(e.target.value)}
                className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 text-gray-300 
                focus:ring-2 focus:ring-emerald-500/50 focus:border-emerald-500/50"
                value=""
              >
                <option value="" className="bg-slate-800">Select team...</option>
                {sortTeamsByDivision(Object.keys(teamDivisions)
                  .filter(team => team !== trade.teamB))
                  .map((team) => (
                    <option key={team} value={team} className="bg-slate-800">
                      {team.toUpperCase()} - {teamDivisions[team].name}
                    </option>
                ))}
              </select>
            )}
          </div>

          {/* Team B */}
          <div className="rounded-xl p-6 border border-slate-700/50 bg-slate-800/50">
            <h3 className="text-lg font-medium text-white mb-4">Select Team 2</h3>
            {trade.teamB ? (
              <div className="flex items-center justify-between rounded-lg px-4 py-3 border border-blue-500/20 bg-slate-700/50">
                <span className="text-blue-400 font-medium">
                  {trade.teamB.toUpperCase()} - {teamDivisions[trade.teamB].name}
                </span>
                <button
                  onClick={() => handleTeamRemove(trade.teamB!)}
                  className="text-blue-400 hover:text-blue-300"
                >
                  ×
                </button>
              </div>
            ) : (
              <select
                onChange={(e) => handleTeamAdd(e.target.value)}
                className="w-full bg-slate-700/50 border border-slate-600 rounded-lg px-4 py-3 text-gray-300 
                focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500/50"
                value=""
              >
                <option value="" className="bg-slate-800">Select team...</option>
                {sortTeamsByDivision(Object.keys(teamDivisions)
                  .filter(team => team !== trade.teamA))
                  .map((team) => (
                    <option key={team} value={team} className="bg-slate-800">
                      {team.toUpperCase()} - {teamDivisions[team].name}
                    </option>
                ))}
              </select>
            )}
          </div>
        </div>
        
        {/* Trade Builder */}
        {trade.teamA && trade.teamB && (
          <div className="space-y-8">
            <div className="grid md:grid-cols-2 gap-8">
              <div className="rounded-xl p-6 border border-slate-700/50 bg-slate-800/50">
                <TeamPlayerList
                  team={trade.teamA}
                  availablePlayers={players.filter(p => p.team?.toLowerCase() === trade.teamB?.toLowerCase())}
                  availableProspects={prospects.filter(p => p.org?.toLowerCase() === trade.teamB?.toLowerCase())}
                  receivingAssets={trade.teamAReceiving}
                  onAssetSelect={(asset, isProspect) => handleAssetAdd(trade.teamA!, asset, isProspect)}
                  onAssetRemove={(asset) => handleAssetRemove(trade.teamA!, asset)}
                  otherTeam={trade.teamB}
                />
              </div>
              <div className="rounded-xl p-6 border border-slate-700/50 bg-slate-800/50">
                <TeamPlayerList
                  team={trade.teamB}
                  availablePlayers={players.filter(p => p.team?.toLowerCase() === trade.teamA?.toLowerCase())}
                  availableProspects={prospects.filter(p => p.org?.toLowerCase() === trade.teamA?.toLowerCase())}
                  receivingAssets={trade.teamBReceiving}
                  onAssetSelect={(asset, isProspect) => handleAssetAdd(trade.teamB!, asset, isProspect)}
                  onAssetRemove={(asset) => handleAssetRemove(trade.teamB!, asset)}
                  otherTeam={trade.teamA}
                />
              </div>
            </div>
            
            {loading && (
              <div className="text-center py-8">
                <div className="inline-block animate-spin rounded-full h-8 w-8 border-4 border-emerald-500 border-t-transparent"></div>
                <p className="mt-2 text-gray-400">Analyzing trade...</p>
              </div>
            )}
            
            {analysis && (
              <div className="rounded-xl p-6 border border-slate-700/50 bg-slate-800/50">
                <ValueDisplay 
                  analysis={analysis} 
                  team1Name={trade.teamA?.toUpperCase() || ''} 
                  team2Name={trade.teamB?.toUpperCase() || ''}
                />
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default TradeAnalyzer;