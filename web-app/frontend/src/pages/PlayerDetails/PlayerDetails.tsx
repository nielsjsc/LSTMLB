import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import { getPlayerDetails, PlayerStats } from '../../services/api';
import { CombinedHittingTable, CombinedPitchingTable } from '../../components/Tables';



const PlayerDetails = () => {
    const { playerId } = useParams<{ playerId: string }>();
    const [player, setPlayer] = useState<PlayerStats | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    useEffect(() => {
        const fetchPlayer = async () => {
            if (!playerId) {
                console.log('No player ID provided');
                return;
            }
            setLoading(true);
            try {
                console.log('Attempting to fetch player:', playerId);
                const data = await getPlayerDetails(parseInt(playerId));
                console.log('Successfully fetched data:', data);
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

    const calculateStats = () => {
        if (!player?.projections) return null;
    
        const calculateTotalWar = (proj: typeof player.projections[0]) => 
            (proj.hitting?.war_bat || 0) + (proj.pitching?.war_pit || 0);
    
        const actual = player.projections
            .filter(p => p.year < 2025)
            .reduce((acc, proj) => ({
                war: acc.war + calculateTotalWar(proj),
                value: acc.value + (proj.value.base_value || 0)
            }), { war: 0, value: 0 });
    
        const projected = player.projections
            .filter(p => p.year >= 2025)
            .reduce((acc, proj) => ({
                war: acc.war + calculateTotalWar(proj),
                value: acc.value + proj.value.base_value
            }), { war: 0, value: 0 });
    
        return {
            actual,
            projected,
            total: {
                war: actual.war + projected.war,
                value: actual.value + projected.value
            }
        };
    };
    const calculateTotalSurplus = () => {
        if (!player?.projections) return 0;
        return player.projections.reduce((sum, proj) => sum + (proj.value.surplus_value || 0), 0);
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
    
    const hasPitchingStats = player?.projections.some(
        proj => proj.pitching?.era != null
    );

    const hasHittingStats = player?.projections.some(
        proj => proj.hitting?.avg != null
    );

    const get2025Data = () => {
        return player?.projections.find(p => p.year === 2025) || player?.projections[0];
    };
    const pitchingTableData = player?.projections
        .sort((a, b) => b.year - a.year)  // Changed sort order
        .filter((proj): proj is (typeof proj & { pitching: NonNullable<typeof proj.pitching> }) => 
            proj.pitching?.war_pit != null
        )
        .map(proj => ({
            year: proj.year,
            age: proj.age,
            value: {
                base_value: proj.value.base_value,
                contract_value: proj.value.contract_value,
                surplus_value: proj.value.surplus_value
            },
            pitching: proj.pitching
        })) || [];

    const hittingTableData = player?.projections
        .sort((a, b) => b.year - a.year)  // Changed sort order
        .filter((proj): proj is (typeof proj & { hitting: NonNullable<typeof proj.hitting> }) => 
            proj.hitting?.war_bat != null
        )
        .map(proj => ({
            year: proj.year,
            age: proj.age,
            value: {
                base_value: proj.value.base_value,
                contract_value: proj.value.contract_value,
                surplus_value: proj.value.surplus_value
            },
            hitting: proj.hitting
        })) || [];


        const stats = calculateStats();
        const formatValue = (value: number) => `$${(value / 1000000).toFixed(1)}M`;



        return (
            <div className="min-h-screen bg-gray-50">
              <div className="max-w-7xl mx-auto py-8 px-4">
                {loading ? (
                  <div className="flex justify-center items-center h-64">
                    <div className="animate-spin rounded-full h-8 w-8 border-4 border-emerald-500 border-t-transparent"></div>
                  </div>
                ) : error ? (
                  <div className="bg-red-50 border-l-4 border-red-400 p-4 rounded">
                    <p className="text-red-700">{error}</p>
                  </div>
                ) : (
                  <>
                    {/* Player Hero Section */}
                    <div className="bg-white shadow-sm rounded-lg p-8 mb-8">
                      <div className="flex justify-between items-start mb-6">
                        <div>
                          <h1 className="text-3xl font-bold text-gray-900 mb-2">{player?.name}</h1>
                          <div className="flex items-center space-x-3 text-gray-600">
                            <span className="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-gray-100">
                              {get2025Data()?.team?.toUpperCase()}
                            </span>
                            <span className="text-sm">{player?.position}</span>
                          </div>
                        </div>
                      </div>
          
                      {/* Stats Grid */}
                      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="bg-gray-50 rounded-lg p-6 border border-gray-100">
                          <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-4">Actual Stats</h3>
                          <div className="space-y-3">
                            <div className="flex justify-between items-center">
                              <span className="text-gray-600">WAR</span>
                              <span className="text-lg font-semibold text-gray-900">{stats?.actual.war.toFixed(1)}</span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-gray-600">Value</span>
                              <span className="text-lg font-semibold text-emerald-600">{formatValue(stats?.actual.value || 0)}</span>
                            </div>
                          </div>
                        </div>
          
                        <div className="bg-gray-50 rounded-lg p-6 border border-gray-100">
                          <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-4">Projected Stats</h3>
                          <div className="space-y-3">
                            <div className="flex justify-between items-center">
                              <span className="text-gray-600">WAR</span>
                              <span className="text-lg font-semibold text-gray-900">{stats?.projected.war.toFixed(1)}</span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-gray-600">Value</span>
                              <span className="text-lg font-semibold text-emerald-600">{formatValue(stats?.projected.value || 0)}</span>
                            </div>
                          </div>
                        </div>
          
                        <div className="bg-gray-50 rounded-lg p-6 border border-gray-100">
                          <h3 className="text-sm font-medium text-gray-500 uppercase tracking-wider mb-4">Career Totals</h3>
                          <div className="space-y-3">
                            <div className="flex justify-between items-center">
                              <span className="text-gray-600">WAR</span>
                              <span className="text-lg font-semibold text-gray-900">{stats?.total.war.toFixed(1)}</span>
                            </div>
                            <div className="flex justify-between items-center">
                              <span className="text-gray-600">Value</span>
                              <span className="text-lg font-semibold text-emerald-600">{formatValue(stats?.total.value || 0)}</span>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
          
                    {/* Projections Tables */}
                    <div className="space-y-8">
                      {hasPitchingStats && (
                        <section className="bg-white shadow-sm rounded-lg p-6">
                          <h2 className="text-xl font-semibold text-gray-900 mb-6">Pitching Projections</h2>
                          <CombinedPitchingTable 
                            data={[...pitchingTableData].sort((a, b) => a.year - b.year)}
                            dividerYear={2025}
                          />
                        </section>
                      )}
          
                      {hasHittingStats && (
                        <section className="bg-white shadow-sm rounded-lg p-6">
                          <h2 className="text-xl font-semibold text-gray-900 mb-6">Hitting Projections</h2>
                          <CombinedHittingTable 
                            data={[...hittingTableData].sort((a, b) => a.year - b.year)}
                            dividerYear={2025}
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