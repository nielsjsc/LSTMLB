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
export interface PlayerFilter {
    year?: number;
    team?: string;
    position?: string;
    sort_by?: 'war' | 'value';
}

export interface PlayerResponse {
    count: number;
    players: Array<{
        name: string;
        team: string;
        position: string;
        war: number;
        base_value: number;
        contract_value: number;
        surplus_value: number;
    }>;
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
                base_value: number;  // Add this field
                contract_value: number;
                surplus_value: number;
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
                base_value: number;  // Add this field
                contract_value: number;
                surplus_value: number;
                status: string;
            }>;
        }>;
    };
}

const API_BASE = 'http://localhost:8000/api';

export const getPlayers = async (year: number = 2024): Promise<Player[]> => {
    try {
        const response = await fetch(`${API_BASE}/players?year=${year}`);
        if (!response.ok) throw new Error('Failed to fetch players');
        const data = await response.json();
        return data.players; // Return just the players array for Trade Analyzer
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

export const filterPlayers = async (filters: PlayerFilter): Promise<PlayerResponse> => {
    const params = new URLSearchParams();
    if (filters.year) params.append('year', filters.year.toString());
    if (filters.team) params.append('team', filters.team);
    if (filters.position) params.append('position', filters.position);
    if (filters.sort_by) params.append('sort_by', filters.sort_by);

    try {
        const response = await fetch(`${API_BASE}/players?${params}`);
        if (!response.ok) throw new Error('Failed to fetch players');
        return response.json();
    } catch (error) {
        console.error('Error fetching players:', error);
        throw error;
    }
};