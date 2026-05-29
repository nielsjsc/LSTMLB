import { useState, useEffect } from 'react';
import { 
  analyzeTrade, 
  Player, 
  Prospect,
  TradeAnalysis 
} from '../../services/api';
import { useTradeAssets } from '../../hooks/useApi';
import { teamDivisions } from '../../config/teams';
import { getTeamColors } from '../../utils/teamColors';
import TeamPlayerList from './components/PlayerSelector/TeamPlayerList';
import ValueDisplay from './components/TradeBreakdown/ValueDisplay';

interface TradeState {
  teamA: string | null;
  teamB: string | null;
  teamAReceiving: Array<{ asset: Player | Prospect; isProspect: boolean }>;
  teamBReceiving: Array<{ asset: Player | Prospect; isProspect: boolean }>;
}

const TradeAnalyzer = () => {
  // React Query — replaces manual Promise.all + useState + useEffect
  const { players, prospects, isLoading: initialLoading, error: assetsError } = useTradeAssets();

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(assetsError?.message ?? null);
  const [analysis, setAnalysis] = useState<TradeAnalysis | null>(null);
  const [trade, setTrade] = useState<TradeState>({
    teamA: null,
    teamB: null,
    teamAReceiving: [],
    teamBReceiving: []
  });

  // Propagate asset-loading errors to the local error state
  useEffect(() => {
    if (assetsError) setError(assetsError.message);
  }, [assetsError]);

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

  const handleResetTrade = () => {
    setTrade({ teamA: null, teamB: null, teamAReceiving: [], teamBReceiving: [] });
    setAnalysis(null);
    setError(null);
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
      setError(errorMessage);
      setAnalysis(null);
    } finally {
      setLoading(false);
    }
  };

  const teamAColors = trade.teamA ? getTeamColors(trade.teamA) : null;
  const teamBColors = trade.teamB ? getTeamColors(trade.teamB) : null;
  const hasPlayers = trade.teamAReceiving.length > 0 || trade.teamBReceiving.length > 0;

  // Team selector component
  const TeamSelector = ({ side, team, otherTeam }: { side: 'A' | 'B'; team: string | null; otherTeam: string | null }) => {
    const colors = team ? getTeamColors(team) : null;
    const label = side === 'A' ? 'Team 1' : 'Team 2';

    if (team) {
      return (
        <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-white group">
          {/* Team color accent bar */}
          <div className="absolute top-0 left-0 right-0 h-1 opacity-80" 
            style={{ background: colors?.gradient }} />
          <div className="px-5 py-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg flex items-center justify-center font-bold text-sm text-gray-900"
                style={{ background: `${colors?.primary}30`, border: `1px solid ${colors?.primary}40` }}>
                {team.toUpperCase().slice(0, 3)}
              </div>
              <div>
                <p className="text-gray-900 font-semibold text-sm">{teamDivisions[team]?.name || team.toUpperCase()}</p>
                <p className="text-gray-400 text-xs">{teamDivisions[team]?.division}</p>
              </div>
            </div>
            <button
              onClick={() => handleTeamRemove(team)}
              className="w-8 h-8 flex items-center justify-center rounded-lg text-gray-400 hover:text-red-400 hover:bg-red-500/10 transition-all duration-200 opacity-0 group-hover:opacity-100"
              title="Remove team"
            >
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      );
    }

    return (
      <div className="rounded-xl border border-dashed border-gray-300 bg-gray-50 p-1">
        <select
          onChange={(e) => handleTeamAdd(e.target.value)}
          className="w-full bg-transparent rounded-lg px-4 py-4 text-gray-500 text-sm font-medium
            focus:ring-2 focus:ring-brand-500/30 focus:border-brand-200 transition-all duration-200 cursor-pointer
            appearance-none"
          value=""
          style={{ backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 20 20' fill='%2394a3b8'%3E%3Cpath fill-rule='evenodd' d='M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z' clip-rule='evenodd'/%3E%3C/svg%3E")`, backgroundRepeat: 'no-repeat', backgroundPosition: 'right 12px center', backgroundSize: '20px' }}
        >
          <option value="" className="bg-white text-gray-500">Select {label}...</option>
          {Object.keys(teamDivisions).filter(t => t !== otherTeam).sort().map((t) => (
            <option key={t} value={t} className="bg-white text-gray-800">
              {t.toUpperCase()} — {teamDivisions[t].name}
            </option>
          ))}
        </select>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-\[#F5F3EE\]">
      <div className="max-w-7xl mx-auto py-8 px-4">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-gray-50 border border-gray-200 mb-4">
            <span className="w-1.5 h-1.5 rounded-full bg-brand-400 animate-pulse" />
            <span className="text-xs text-gray-500 font-medium tracking-wider uppercase">Live Analysis</span>
          </div>
          <h1 className="text-3xl font-bold mb-3 bg-clip-text text-transparent bg-gradient-brand pb-1 font-display tracking-tight">
            Trade Simulator
          </h1>
          <p className="text-gray-500 max-w-xl mx-auto text-sm leading-relaxed">
            Build trades between any two teams. Player values update in real-time using our projection-based surplus valuations and prospect rankings.
          </p>
        </div>

        {/* Initial loading state */}
        {initialLoading && (
          <div className="flex flex-col items-center justify-center py-20">
            <div className="relative">
              <div className="w-12 h-12 rounded-full border-2 border-gray-200 border-t-brand-400 animate-spin" />
              <div className="absolute inset-0 w-12 h-12 rounded-full border-2 border-transparent border-b-accent-blue/40 animate-spin" style={{ animationDirection: 'reverse', animationDuration: '1.5s' }} />
            </div>
            <p className="mt-4 text-gray-500 text-sm">Loading player data...</p>
          </div>
        )}

        {!initialLoading && (
          <>
            {error && (
              <div className="flex items-center gap-3 border border-red-500/20 rounded-xl px-5 py-4 mb-6 bg-red-500/5">
                <svg className="w-5 h-5 text-red-400 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                  <path strokeLinecap="round" strokeLinejoin="round" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
                </svg>
                <p className="text-red-400 text-sm">{error}</p>
              </div>
            )}

            {/* Team Selection Row */}
            <div className="grid md:grid-cols-[1fr,auto,1fr] gap-4 items-center mb-8">
              <TeamSelector side="A" team={trade.teamA} otherTeam={trade.teamB} />
              
              {/* VS Divider */}
              <div className="hidden md:flex flex-col items-center gap-2">
                <div className="w-12 h-12 rounded-full bg-white border border-gray-200 flex items-center justify-center shadow-lg">
                  <span className="text-sm font-bold text-gray-500 tracking-wider">VS</span>
                </div>
                {(trade.teamA || trade.teamB) && (
                  <button
                    onClick={handleResetTrade}
                    className="text-[10px] text-gray-400 hover:text-red-400 transition-colors duration-200 font-medium uppercase tracking-wider"
                  >
                    Reset
                  </button>
                )}
              </div>

              <TeamSelector side="B" team={trade.teamB} otherTeam={trade.teamA} />
            </div>

            {/* Mobile VS + Reset */}
            <div className="md:hidden flex items-center justify-center gap-4 -mt-2 mb-6">
              <div className="w-10 h-10 rounded-full bg-white border border-gray-200 flex items-center justify-center">
                <span className="text-xs font-bold text-gray-500">VS</span>
              </div>
              {(trade.teamA || trade.teamB) && (
                <button onClick={handleResetTrade} className="text-xs text-gray-400 hover:text-red-400 transition-colors">
                  Reset Trade
                </button>
              )}
            </div>

            {/* Trade Builder */}
            {trade.teamA && trade.teamB && (
              <div className="space-y-6">
                {/* Player selection panels */}
                <div className="grid md:grid-cols-2 gap-4">
                  {/* Team A panel */}
                  <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-gray-50">
                    <div className="absolute top-0 left-0 right-0 h-0.5 opacity-60"
                      style={{ background: teamAColors?.gradient }} />
                    <div className="p-5">
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
                  </div>

                  {/* Team B panel */}
                  <div className="relative rounded-xl overflow-hidden border border-gray-200 bg-gray-50">
                    <div className="absolute top-0 left-0 right-0 h-0.5 opacity-60"
                      style={{ background: teamBColors?.gradient }} />
                    <div className="p-5">
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
                </div>
                
                {/* Loading indicator */}
                {loading && (
                  <div className="flex items-center justify-center gap-3 py-6">
                    <div className="relative">
                      <div className="w-8 h-8 rounded-full border-2 border-gray-200 border-t-brand-400 animate-spin" />
                    </div>
                    <p className="text-gray-500 text-sm font-medium">Analyzing trade value...</p>
                  </div>
                )}
                
                {/* Analysis Results */}
                {analysis && !loading && (
                  <div className="rounded-xl border border-gray-200 bg-gray-50 p-6">
                    <div className="flex items-center gap-2 mb-5">
                      <svg className="w-5 h-5 text-brand-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                      </svg>
                      <h2 className="text-gray-900 font-semibold">Trade Analysis</h2>
                    </div>
                    <ValueDisplay 
                      analysis={analysis} 
                      team1Name={trade.teamA?.toUpperCase() || ''} 
                      team2Name={trade.teamB?.toUpperCase() || ''}
                    />
                  </div>
                )}

                {/* Empty state prompt */}
                {!analysis && !loading && !hasPlayers && (
                  <div className="flex flex-col items-center justify-center py-12 rounded-xl border border-dashed border-gray-200">
                    <svg className="w-16 h-16 text-gray-600 mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1}>
                      <path strokeLinecap="round" strokeLinejoin="round" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                    </svg>
                    <p className="text-gray-400 text-sm font-medium mb-1">Add players to build your trade</p>
                    <p className="text-gray-400 text-xs">Search for MLB players or prospects on each side</p>
                  </div>
                )}
              </div>
            )}

            {/* Empty state: no teams selected */}
            {!trade.teamA && !trade.teamB && (
              <div className="flex flex-col items-center justify-center py-20">
                <div className="w-20 h-20 rounded-2xl bg-white/80 border border-gray-200 flex items-center justify-center mb-6">
                  <svg className="w-10 h-10 text-gray-500" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
                  </svg>
                </div>
                <h3 className="text-gray-900 font-semibold mb-2">Select Two Teams</h3>
                <p className="text-gray-400 text-sm text-center max-w-sm">
                  Choose two teams above to start building a trade. Player values are based on projected performance and contract surplus.
                </p>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default TradeAnalyzer;