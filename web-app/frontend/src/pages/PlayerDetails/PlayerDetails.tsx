import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { getPlayerDetails, PlayerStats } from '../../services/api';import { CURRENT_YEAR } from '../../config';import { CombinedHittingTable, CombinedPitchingTable } from '../../components/Tables';

const StatCard: React.FC<{
  title: string;
  stats: { war: number; value: number; surplus?: number; contract?: number };
  showSurplus?: boolean;
}> = ({ title, stats, showSurplus = false }) => {
  const formatValue = (value: number) => `$${(value / 1000000).toFixed(1)}M`;
  
  return (
    <div className="rounded-xl p-6 border border-slate-700/50 bg-slate-800/50">
      <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-4">{title}</h3>
      <div className="space-y-4">
        <div className="flex justify-between items-center">
          <span className="text-gray-400">WAR</span>
          <span className={`text-lg font-semibold ${stats.war > 0 ? 'text-emerald-400' : 'text-gray-500'}`}>
            {stats.war.toFixed(1)}
          </span>
        </div>
        <div className="flex justify-between items-center">
          <span className="text-gray-400">Value</span>
          <span className={`text-lg font-semibold ${stats.value >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
            {formatValue(stats.value)}
          </span>
        </div>
        {showSurplus && stats.contract !== undefined && (
          <div className="flex justify-between items-center">
            <span className="text-gray-400">Contract</span>
            <span className="text-lg font-semibold text-red-400">
              {formatValue(stats.contract)}
            </span>
          </div>
        )}
        {showSurplus && stats.surplus !== undefined && (
          <div className="flex justify-between items-center">
            <span className="text-gray-400">Surplus</span>
            <span className={`text-lg font-semibold ${stats.surplus >= 0 ? 'text-emerald-400' : 'text-red-400'}`}>
              {formatValue(stats.surplus)}
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

const PlayerDetails = () => {
  const { playerId } = useParams<{ playerId: string }>();
  const [player, setPlayer] = useState<PlayerStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPlayer = async () => {
      if (!playerId) return;
      setLoading(true);
      try {
        const data = await getPlayerDetails(parseInt(playerId));
        setPlayer(data);
        setError(null);
      } catch (err) {
        console.error('Error fetching player:', err);
        setError('Failed to load player details');
      } finally {
        setLoading(false);
      }
    };
    fetchPlayer();
  }, [playerId]);

  const hasCurrentYearStats = (data: PlayerStats['projections'] | undefined, type: 'hitting' | 'pitching'): boolean => {
    if (!data) return false;
    return data.some(proj => 
      proj.year === CURRENT_YEAR && 
      (type === 'hitting' ? proj.hitting?.war_bat != null : proj.pitching?.war_pit != null)
    );
  };
  
  const getFAYears = () => {
    if (!player?.projections[0]) return { earliest: '-', probable: '-', latest: '-' };
    const firstYear = player.projections[0];
    return {
      earliest: firstYear.earliest_fa_year || '-',
      probable: firstYear.probable_fa_year || '-',
      latest: firstYear.fa_year || '-'
    };
  };

  const hasPitchingStats = player?.projections.some(proj => proj.pitching?.era != null);
  const hasHittingStats = player?.projections.some(proj => proj.hitting?.avg != null);
  const getCurrentYearData = () => player?.projections.find(p => p.year === CURRENT_YEAR) || player?.projections[0];
  const dataCurrentYear = getCurrentYearData();
  
  const pitchingTableData = React.useMemo(() => {
    if (!player?.projections) return [];
    return player.projections
      .filter((proj): proj is (typeof proj & { pitching: NonNullable<typeof proj.pitching> }) => 
        proj.pitching?.war_pit != null
      )
      .map(proj => ({
        year: proj.year,
        age: proj.age,
        status: proj.status,
        value: proj.value,
        pitching: proj.pitching
      }));
  }, [player]);
  
  const hittingTableData = React.useMemo(() => {
    if (!player?.projections) return [];
    return player.projections
      .filter((proj): proj is (typeof proj & { hitting: NonNullable<typeof proj.hitting> }) => 
        proj.hitting?.war_bat != null
      )
      .map(proj => ({
        year: proj.year,
        age: proj.age,
        status: proj.status,
        value: proj.value,
        hitting: proj.hitting
      }));
  }, [player]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      <div className="max-w-7xl mx-auto py-8 px-4">
        {loading ? (
          <div className="flex justify-center items-center h-64">
            <div className="animate-spin rounded-full h-8 w-8 border-4 border-emerald-400 border-t-transparent"></div>
          </div>
        ) : error ? (
          <div className="rounded-lg px-4 py-2 border border-red-500/20 bg-red-500/10">
            <p className="text-red-400">{error}</p>
          </div>
        ) : (
          <>
            <div className="rounded-xl p-8 border border-slate-700/50 bg-slate-800/50 mb-8">
              <div className="flex flex-col md:flex-row justify-between items-start gap-6">
                <div>
                  <div className="flex items-center gap-4 mb-4">
                    <h1 className="text-3xl font-bold text-white">{player?.name}</h1>
                    <div className="flex gap-2">
                      <span className="px-3 py-1 rounded-full text-sm font-medium bg-emerald-400/10 text-emerald-400">
                        {getCurrentYearData()?.team?.toUpperCase()}
                      </span>
                      <span className="px-3 py-1 rounded-full text-sm font-medium bg-slate-700/50 text-gray-300">
                        {player?.position}
                      </span>
                    </div>
                  </div>
                  <div className="text-sm text-gray-400">
                    FA Years: {getFAYears().earliest} (Early) - {getFAYears().probable} (Probable) - {getFAYears().latest} (Late)
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mt-8">
              <div className="rounded-xl p-6 border border-slate-700/50 bg-slate-800/50">
                <h3 className="text-sm font-medium text-gray-400 uppercase tracking-wider mb-2">Trade Value</h3>
                <p className={`text-lg font-semibold ${
                  (dataCurrentYear?.value?.trade_value || 0) >= 0 ? 'text-emerald-400' : 'text-red-400'
                }`}>
                  {`$${((dataCurrentYear?.value?.trade_value || 0) / 1000000).toFixed(1)}M`}
                </p>
              </div>

                <StatCard 
                  title="Historical Stats"
                  stats={{
                    war: dataCurrentYear?.value.historical_war || 0,
                    value: dataCurrentYear?.value.historical_value || 0
                  }}
                />
                <StatCard
                  title="Projected Stats While Under Contract"
                  stats={{
                    war: dataCurrentYear?.value.contract_war || 0,
                    value: dataCurrentYear?.value.contract_base_value || 0,
                    contract: dataCurrentYear?.value.total_contract || 0,
                    surplus: dataCurrentYear?.value.trade_value || 0
                  }}
                  showSurplus={true}
                />
                <StatCard
                  title="Total Value"
                  stats={{
                    war: dataCurrentYear?.value.total_war || 0,
                    value: dataCurrentYear?.value.total_value || 0
                  }}
                />
              </div>
            </div>

            <div className="space-y-8">
              {player && hasPitchingStats && hasCurrentYearStats(player.projections, 'pitching') && (
                <section className="rounded-xl overflow-hidden border border-slate-700/50 bg-slate-800/50">
                  <div className="p-6 border-b border-slate-700/50">
                    <h2 className="text-xl font-semibold text-white">Pitching Stats</h2>
                  </div>
                  <CombinedPitchingTable 
                    data={pitchingTableData.sort((a, b) => b.year - a.year)}
                  />
                </section>
              )}

              {player && hasHittingStats && hasCurrentYearStats(player.projections, 'hitting') && (
                <section className="rounded-xl overflow-hidden border border-slate-700/50 bg-slate-800/50">
                  <div className="p-6 border-b border-slate-700/50">
                    <h2 className="text-xl font-semibold text-white">Hitting Stats</h2>
                  </div>
                  <CombinedHittingTable 
                    data={hittingTableData.sort((a, b) => b.year - a.year)}
                  />
                </section>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
};

export default PlayerDetails;
