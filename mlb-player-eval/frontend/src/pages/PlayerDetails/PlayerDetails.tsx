import React, { useState, useEffect } from 'react'
import { useParams } from 'react-router-dom'
import { getPlayerDetails, PlayerStats } from '../../services/api'

const PlayerDetails = () => {
  const { playerName } = useParams<{ playerName: string }>()
  const [player, setPlayer] = useState<PlayerStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetchPlayer = async () => {
      if (!playerName) return
      setLoading(true)
      try {
        console.log('Fetching player:', playerName)
        const data = await getPlayerDetails(playerName)
        console.log('Player data:', data)
        setPlayer(data)
        setError(null)
      } catch (err) {
        console.error('Error:', err)
        setError('Failed to load player details')
      } finally {
        setLoading(false)
      }
    }
    fetchPlayer()
  }, [playerName])

  const formatValue = (value: number) => `$${(value / 1000000).toFixed(1)}M`
  const formatPercent = (value: number) => `${(value * 100).toFixed(1)}%`
  const formatDecimal = (value: number) => value.toFixed(3)

  if (loading) return <div className="p-4">Loading...</div>
  if (error) return <div className="p-4 text-red-600">{error}</div>
  if (!player) return <div className="p-4">Player not found</div>

  return (
    <div className="max-w-4xl mx-auto py-8">
        <h1 className="text-3xl font-bold mb-2">{player.name}</h1>
        <p className="text-xl text-gray-600 mb-6">
            {player.team.toUpperCase()} - {player.position || 'Unknown'}
        </p>
        
        <div className="space-y-6">
            {player.projections.map(year => (
                <div key={year.year} className="border rounded-lg p-4">
                    <h2 className="text-xl font-semibold mb-4">{year.year} Projections</h2>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div className="space-y-2">
                            <h3 className="font-medium">Value Metrics</h3>
                            <p>WAR: {year.war.toFixed(1)}</p>
                            <p>Base Value: ${(year.value.base/1000000).toFixed(1)}M</p>
                            <p>Contract: ${(year.value.contract/1000000).toFixed(1)}M</p>
                            <p>Surplus: ${(year.value.surplus/1000000).toFixed(1)}M</p>
                        </div>

                        {year.hitting && Object.values(year.hitting).some(v => v !== null) && (
                            <div className="space-y-2">
                                <h3 className="font-medium">Hitting Projections</h3>
                                {year.hitting.age && <p>Age: {year.hitting.age}</p>}
                                
                                {/* Plate Discipline */}
                                {year.hitting.bb_pct && <p>BB%: {formatPercent(year.hitting.bb_pct)}</p>}
                                {year.hitting.k_pct && <p>K%: {formatPercent(year.hitting.k_pct)}</p>}
                                
                                {/* Triple Slash */}
                                <div className="flex space-x-4">
                                    {year.hitting.avg && <p>AVG: {formatDecimal(year.hitting.avg)}</p>}
                                    {year.hitting.obp && <p>OBP: {formatDecimal(year.hitting.obp)}</p>}
                                    {year.hitting.slg && <p>SLG: {formatDecimal(year.hitting.slg)}</p>}
                                </div>
                                
                                {/* Advanced Stats */}
                                {year.hitting.woba && <p>wOBA: {formatDecimal(year.hitting.woba)}</p>}
                                {year.hitting.wrc_plus && <p>wRC+: {Math.round(year.hitting.wrc_plus)}</p>}
                                {year.hitting.ev && <p>Exit Velocity: {year.hitting.ev.toFixed(1)}</p>}
                                
                                {/* Value Stats */}
                                {year.hitting.off && <p>Offensive Value: {year.hitting.off.toFixed(1)}</p>}
                                {year.hitting.bsr && <p>Baserunning: {year.hitting.bsr.toFixed(1)}</p>}
                                {year.hitting.def && <p>Defensive Value: {year.hitting.def.toFixed(1)}</p>}
                            </div>
                        )}

                        {year.pitching && Object.values(year.pitching).some(v => v !== null) && (
                            <div className="space-y-2">
                                <h3 className="font-medium">Pitching Projections</h3>
                                {year.pitching.age && <p>Age: {year.pitching.age}</p>}
                                
                                {/* Core Stats */}
                                {year.pitching.fip && <p>FIP: {year.pitching.fip.toFixed(2)}</p>}
                                {year.pitching.siera && <p>SIERA: {year.pitching.siera.toFixed(2)}</p>}
                                
                                {/* Plate Discipline */}
                                {year.pitching.k_pct && <p>K%: {formatPercent(year.pitching.k_pct)}</p>}
                                {year.pitching.bb_pct && <p>BB%: {formatPercent(year.pitching.bb_pct)}</p>}
                                
                                {/* Batted Ball */}
                                {year.pitching.gb_pct && <p>GB%: {formatPercent(year.pitching.gb_pct)}</p>}
                                {year.pitching.fb_pct && <p>FB%: {formatPercent(year.pitching.fb_pct)}</p>}
                                
                                {/* Plus Stats */}
                                {year.pitching.stuff_plus && <p>Stuff+: {Math.round(year.pitching.stuff_plus)}</p>}
                                {year.pitching.location_plus && <p>Location+: {Math.round(year.pitching.location_plus)}</p>}
                                {year.pitching.pitching_plus && <p>Pitching+: {Math.round(year.pitching.pitching_plus)}</p>}
                                
                                {/* Velocity */}
                                {year.pitching.fbv && <p>Fastball Velocity: {year.pitching.fbv.toFixed(1)}</p>}
                            </div>
                        )}
                    </div>
                </div>
            ))}
        </div>
    </div>
);
}

export default PlayerDetails