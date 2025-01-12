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

    const formatValue = (value: number | undefined) => 
        value ? `$${(value / 1000000).toFixed(1)}M` : '-';
    const formatPercent = (value: number | undefined) => 
        value ? `${(value * 100).toFixed(1)}%` : '-';
    const formatDecimal = (value: number | undefined) => 
        value ? value.toFixed(3) : '-';

    const hasPitchingStats = player?.projections.some(
        proj => proj.pitching?.war_pit != null
    );

    const hasHittingStats = player?.projections.some(
        proj => proj.hitting?.war_bat != null
    );

    const pitchingTableData = player?.projections
        .filter((proj): proj is (typeof proj & { pitching: NonNullable<typeof proj.pitching> }) => 
            proj.pitching?.war_pit != null
        )
        .map(proj => ({
            year: proj.year,
            age: proj.age,
            value: proj.value,
            pitching: proj.pitching
        })) || [];

    const hittingTableData = player?.projections
        .filter((proj): proj is (typeof proj & { hitting: NonNullable<typeof proj.hitting> }) => 
            proj.hitting?.war_bat != null
        )
        .map(proj => ({
            year: proj.year,
            age: proj.age,
            value: proj.value,
            hitting: proj.hitting
        })) || [];
    console.log('Has pitching stats:', hasPitchingStats);
    console.log('Pitching data:', pitchingTableData);
    console.log('Has hitting stats:', hasHittingStats);
    console.log('Hitting data:', hittingTableData);
    
    return (
        <div className="max-w-7xl mx-auto py-8 px-4">
            <div className="mb-8">
                <h1 className="text-3xl font-bold">{player?.name}</h1>
                <p className="text-gray-600">{player?.team.toUpperCase()} - {player?.position}</p>
            </div>

            <div className="space-y-8">
                {hasPitchingStats && (
                    <section>
                        <h2 className="text-xl font-semibold mb-4">Pitching Projections</h2>
                        <CombinedPitchingTable data={pitchingTableData} />
                    </section>
                )}

                {hasHittingStats && (
                    <section>
                        <h2 className="text-xl font-semibold mb-4">Hitting Projections</h2>
                        <CombinedHittingTable data={hittingTableData} />
                    </section>
                )}
            </div>
        </div>
    );
};
    
    export default PlayerDetails;