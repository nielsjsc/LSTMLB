export interface Player {
    name: string;
    team: string;
    status: string | null;
    year: number;
    war: number;
    base_value: number;
    contract_value: number;
    surplus_value: number;
}

export interface TradeAnalysis {
    team1: {
        total_value: number;
        players: Array<{
            name: string;
            team: string;
            status: string;
            total_surplus: number;
            total_contract: number;
            yearly_projections: Array<{
                year: number;
                war: number;
                surplus_value: number;
                contract_value: number;
                status: string;
            }>;
        }>;
    };
    team2: {
        total_value: number;
        players: Array<{
            name: string;
            team: string;
            status: string;
            total_surplus: number;
            total_contract: number;
            yearly_projections: Array<{
                year: number;
                war: number;
                surplus_value: number;
                contract_value: number;
                status: string;
            }>;
        }>;
    };
}

const API_BASE = 'http://localhost:8000/api';

export const getPlayers = async (year: number = 2024): Promise<Player[]> => {
    try {
        const response = await fetch(`${API_BASE}/players?year=${year}`);
        console.log('API Response Status:', response.status);
        const data = await response.json();
        console.log('Raw API Response:', data); // Debug full response
        return data;
    } catch (error) {
        console.error('Error fetching players:', error);
        throw error;
    }
};


export const analyzeTrade = async (
    team1Players: string[],
    team2Players: string[]
): Promise<TradeAnalysis> => {
    try {
        const response = await fetch(`${API_BASE}/trades/analyze`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ team1_players: team1Players, team2_players: team2Players })
        });
        if (!response.ok) throw new Error('Failed to analyze trade');
        return response.json();
    } catch (error) {
        console.error('Error analyzing trade:', error);
        throw error;
    }
};