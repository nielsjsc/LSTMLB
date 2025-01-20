export interface Player {
    id: number;
    name: string;
    team: string;
    position: string;
    status: string | null;
    year: number;
    war_bat?: number | null;  // Changed from war
    war_pit?: number | null;  // Added
    base_value: number;
    contract_value: number;
    surplus_value: number;
}
export interface PlayerFilter {
    year?: number;
    team?: string;
    position?: string;
    sort_by?: 'war' | 'value';
    search?: string; 
}



export interface PlayerResponse {
    count: number;
    players: Array<{
        id: number;
        name: string;
        team: string;
        position: string;
        war_bat?: number | null;  // Changed
        war_pit?: number | null;  // Added
        base_value: number;
        contract_value: number;
        surplus_value: number;
    }>;
}
export interface TradeAnalysis {
    team1: {
        total_surplus: number;
        total_contract: number;
        total_production: number;
        players: Array<{
            name: string;
            team: string;
            status: string;
            total_surplus: number;
            total_contract: number;
            total_production: number;
            yearly_projections: Array<{
                year: number;
                war: number;
                base_value: number;
                contract_value: number;
                surplus_value: number;
                status: string;
            }>;
        }>;
    };
    team2: {
        total_surplus: number;
        total_contract: number;
        total_production: number;
        players: Array<{
            name: string;
            team: string;
            status: string;
            total_surplus: number;
            total_contract: number;
            total_production: number;
            yearly_projections: Array<{
                year: number;
                war: number;
                base_value: number;
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
    if (filters.search) params.append('search', filters.search);  // Add search handling

    try {
        const response = await fetch(`${API_BASE}/players/?${params}`);
        if (!response.ok) throw new Error('Failed to fetch players');
        return response.json();
    } catch (error) {
        console.error('Error fetching players:', error);
        throw error;
    }
};

export interface PlayerStats {
    name: string;
    team: string;
    position: string;
    projections: Array<{
        year: number;
        age: number;
        position: string;
        fa_year: number;
        team: string;
        probable_fa_year: number;
        earliest_fa_year: number;
        value: {
            base_value: number;
            contract_value: number;
            surplus_value: number;
        };
        hitting?: {
            war_bat: number;
            bb_pct_bat: number;
            k_pct_bat: number;
            avg: number;
            obp: number;
            slg: number;
            ops: number;
            woba: number;
            wrc_plus: number;
            ev: number;
            off: number;
            bsr: number;
            def_value: number;
            hr: number;
            doubles: number;
            triples: number;
            r: number;
            rbi: number;
            sb: number;
            cs: number;
        };
        pitching?: {
            war_pit: number;
            era: number;
            fip: number;
            siera: number;
            k_pct_pit: number;
            bb_pct_pit: number;
        };
        
    }>;
}
  
  export const getPlayerDetails = async (playerId: number): Promise<PlayerStats> => {
    const url = `${API_BASE}/players/${playerId}/details`;
    console.log('Calling API:', url);
    const response = await fetch(url);
    if (!response.ok) {
        console.error(`API Error: ${response.status}`);
        throw new Error('Failed to fetch player details');
    }
    const data = await response.json();
    console.log('API Response:', data);
    return data;
};

export type BasePosition = 'C' | '1B' | '2B' | '3B' | 'SS' | 'OF' | 'LF' | 'CF' | 'RF' | 'DH' | 'SP' | 'RP';
export type PlayerPosition = string;
export interface ProjectionResponse {
    count: number;
    players: Array<{
        id: number;
        name: string;
        team: string;
        position: string;
        age: number;
        fa_year: number;
        probable_fa_year: number;
        earliest_fa_year: number;
        value: {
            base_value: number;
            contract_value: number;
            surplus_value: number;
        };
        hitting?: {
            war_bat: number;
            bb_pct_bat: number;
            k_pct_bat: number;
            avg: number;
            obp: number;
            slg: number;
            ops: number;
            woba: number;
            wrc_plus: number;
            ev: number;
            off: number;
            bsr: number;
            def_val: number;
            hr: number;
            doubles: number;
            triples: number;
            r: number;
            rbi: number;
            sb: number;
            cs: number;
        };
        pitching?: {
            war_pit: number;
            era: number;
            fip: number;
            siera: number;
            k_pct_pit: number;
            bb_pct_pit: number;
            gb_pct: number;
            fb_pct: number;
            stuff_plus: number;
            location_plus: number;
            pitching_plus: number;
            fbv: number;
        };
    }>;
}

export const getProjections = async (
    year: number,
    playerType: 'hitter' | 'pitcher',
    team?: string,
    position?: PlayerPosition
): Promise<ProjectionResponse> => {
    try {
        const params = new URLSearchParams({
            year: year.toString(),
            player_type: playerType,
            ...(team && { team }),
            ...(position && { position })
        });

        const response = await fetch(`${API_BASE}/projections?${params}`);
        if (!response.ok) {
            throw new Error('Failed to fetch projections');
        }
        const data = await response.json();
        console.log('Projections API Response:', data);
        return data;
    } catch (error) {
        console.error('Error fetching projections:', error);
        throw error;
    }
};