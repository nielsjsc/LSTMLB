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
    trade_value: number;
    age: number | null;
    fa_year: number | null;
    probable_fa_year: number | null;
    earliest_fa_year: number | null;
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
        real_id: number;
        name: string;
        team: string;
        position: string;
        war_bat?: number | null;  // Changed
        war_pit?: number | null;  // Added
        base_value: number;
        contract_value: number;
        surplus_value: number;
        trade_value: number;
        status: string | null;
    }>;
}


export const getAllProspects = async (
  playerType: 'hitter' | 'pitcher',
  year: number = 2025
): Promise<Prospect[]> => {
  try {
    const response = await fetch(`${API_BASE}/trades/prospects?player_type=${playerType}&year=${year}`);
    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Failed to fetch all prospects: ${JSON.stringify(errorData.detail)}`);
    }
    const data = await response.json();
    // Add debug logging
    console.log('Sample prospect data:', data.players[0]);
    return data.players;
  } catch (error) {
    console.error('Error fetching prospects:', error);
    throw error;
  }
};
export interface BaseTradeAsset {
    name: string;
    total_surplus: number;
  }
  
  export interface PlayerTradeAsset extends BaseTradeAsset {
    type: 'player';
    war: number;
    total_production: number;
    total_contract: number;
  }
  
  export interface ProspectTradeAsset extends BaseTradeAsset {
    type: 'prospect';
    value: number;
    fv: string;
    position: string;
  }
  
  export type TradeAsset = PlayerTradeAsset | ProspectTradeAsset;
  
  // Update TradeAnalysis interface to use TradeAsset
  export interface TradeAnalysis {
    team1: {
      total_surplus: number;
      total_contract: number;
      total_production: number;
      assets: TradeAsset[];
    };
    team2: {
      total_surplus: number;
      total_contract: number;
      total_production: number;
      assets: TradeAsset[];
    };
  }
// Update TradeAssetRequest to include type information
export interface TradeAssetRequest {
  name: string;
  isProspect: boolean;
  team: string;
  type?: 'player' | 'prospect';  // Add optional type field
}
  
  export interface TradeRequest {
    team1_assets: TradeAssetRequest[];
    team2_assets: TradeAssetRequest[];
  }
  
  export const analyzeTrade = async (
    team1Assets: TradeAssetRequest[],
    team2Assets: TradeAssetRequest[]
  ): Promise<TradeAnalysis> => {
    try {
      const response = await fetch(`${API_BASE}/trades/analyze`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          team1_assets: team1Assets,
          team2_assets: team2Assets
        })
      });
  
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Trade Analysis Error:', errorData);
        const errorMessage = typeof errorData.detail === 'object' 
          ? JSON.stringify(errorData.detail) 
          : errorData.detail || 'Unknown error';
        throw new Error(`Failed to analyze trade: ${errorMessage}`);
      }
  
      const data = await response.json();
      console.log('Trade Analysis Response:', data);
      return data;
    } catch (error) {
      console.error('Trade analysis error details:', error);
      throw error;
    }
  };
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



export const filterPlayers = async (filters: PlayerFilter): Promise<PlayerResponse> => {
  const params = new URLSearchParams();
  
  // Add all filter parameters
  if (filters.year) params.append('year', filters.year.toString());
  if (filters.team) params.append('team', filters.team);
  if (filters.position) params.append('position', filters.position);
  if (filters.sort_by) params.append('sort_by', filters.sort_by);
  if (filters.search) params.append('search', filters.search);

  const url = `${API_BASE}/players?${params}`;
  console.log('Fetching players with URL:', url); // Debug log

  try {
      const response = await fetch(url);
      
      if (!response.ok) {
          const errorText = await response.text();
          console.error(`Player search failed (${response.status}):`, errorText);
          throw new Error('Failed to fetch players');
      }
      
      const data = await response.json();
      console.log('Search response:', data); // Debug log
      return data;
  } catch (error) {
      console.error('Error in filterPlayers:', error);
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
        status: string;
        position: string;
        fa_year: number;
        team: string;
        probable_fa_year: number;
        earliest_fa_year: number;
        value: {
            base_value: number;
            contract_value: number;
            surplus_value: number;
            trade_value: number;
            contract_war: number;
            avg_war: number;
            total_contract: number;
            avg_contract: number;
            years_control: number;
            control_through: number;
            total_future_war: number;
            total_future_value: number;
            total_war: number;
            total_value: number;
            historical_war: number;
            historical_value: number;
            contract_base_value: number;
        };
        hitting?: {
            g_bat: number;
            war_bat: number;
            bb_pct_bat: number;
            k_pct_bat: number;
            avg: number;
            obp: number;
            slg: number;
            ops: number;
            woba: number;
            wrc_plus: number;
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
            g_pit: number;
            gs: number;
            war_pit: number;
            era: number;
            fip: number;
            siera: number;
            k_pct_pit: number;
            bb_pct_pit: number;
        };
        
    }>;
}
  
export const getPlayerDetails = async (real_id: number): Promise<PlayerStats> => {
  console.log('Starting API call with real_id:', real_id);  // Debug log
  const url = `${API_BASE}/players/${real_id}/details`;
  console.log('Full URL:', url);  // Debug log
  
  try {
      const response = await fetch(url);
      if (!response.ok) {
          const errorText = await response.text();  // Get error details
          console.error(`API Error ${response.status}:`, errorText);
          throw new Error('Failed to fetch player details');
      }
      const data = await response.json();
      console.log('API Response data:', data);  // Debug log
      return data;
  } catch (error) {
      console.error('Detailed error:', error);
      throw error;
  }
};

export type BasePosition = 'C' | '1B' | '2B' | '3B' | 'SS' | 'OF' | 'LF' | 'CF' | 'RF' | 'DH' | 'SP' | 'RP';
export type PlayerPosition = string;
export interface ProjectionResponse {
    count: number;
    players: Array<{
        real_id: number;
        name: string;
        team: string;
        position: string;
        age: number;
        fa_year: number;
        probable_fa_year: number;
        earliest_fa_year: number;
        status: string;
        value: {
            base_value: number;
            contract_value: number;
            surplus_value: number;
        };
        hitting?: {
            g_bat: number;
            war_bat: number;
            bb_pct_bat: number;
            k_pct_bat: number;
            avg: number;
            obp: number;
            slg: number;
            ops: number;
            woba: number;
            wrc_plus: number;
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
            g_pit: number;
            gs: number;
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
    total_count: number;
    page: number;
    page_size: number;
    total_pages: number;
}


  
export interface Prospect {
    playerId: string;
    IDfg: number | null;
    name: string;
    has_mlb: boolean;
    org: string;
    position: string;
    age: number | null;
    level: string | null;
    fv: string;
    value: number | null;
    composite: number | null;

    // Hitter Tool Grades
    hit?: string;
    game?: string;
    raw?: string;
    speed?: string;

    // Pitcher Tool Grades
    fastball?: string;
    slider?: string;
    curve?: string;
    change?: string;
    command?: string;
  }
  
  export interface ProspectResponse {
    count: number;
    page: number;
    pages: number;
    players: Prospect[];
  }
  
  export interface ProspectResponse {
    count: number;
    page: number;
    pages: number;
    players: Prospect[];
  }
  
  export const getProspects = async (
    playerType: 'hitter' | 'pitcher',
    team?: string,
    position?: string,
    year: number = 2025,
    page: number = 1,
    pageSize: number = 25,
    sortBy?: string,
    sortDirection: 'asc' | 'desc' = 'asc'
  ): Promise<ProspectResponse> => {
    const params = new URLSearchParams();
    params.append('player_type', playerType);
    params.append('year', year.toString());
    params.append('page', page.toString());
    params.append('page_size', pageSize.toString());
    if (team) params.append('team', team);
    if (position) params.append('position', position);
    if (sortBy) params.append('sort_by', sortBy);
    params.append('sort_direction', sortDirection);
    
    try {
      const response = await fetch(`${API_BASE}/prospects?${params}`);
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Prospect API Error:', errorData);
        throw new Error(`Failed to fetch prospects: ${errorData.detail || 'Unknown error'}`);
      }
      return response.json();
    } catch (error) {
      console.error('Error fetching prospects:', error);
      throw error;
    }
  };

  export const getProjections = async (
    year: number,
    playerType: 'hitter' | 'pitcher',
    team?: string,
    position?: string,
    page: number = 1,
    pageSize: number = 50,
    sortBy?: string,
    sortDirection: 'asc' | 'desc' = 'desc'
): Promise<ProjectionResponse> => {
    const params = new URLSearchParams({
        year: year.toString(),
        player_type: playerType,
        page: page.toString(),
        page_size: pageSize.toString()
    });

    if (team) params.append('team', team);
    if (position) params.append('position', position);
    if (sortBy) params.append('sort_by', sortBy);
    if (sortDirection) params.append('sort_direction', sortDirection);

    console.log('Fetching projections with params:', params.toString());  // Debug log

    const response = await fetch(`${API_BASE}/projections?${params}`);
    if (!response.ok) {
        const errorText = await response.text();
        console.error(`Projections API Error ${response.status}:`, errorText);
        throw new Error('Failed to fetch projections');
    }
    const data = await response.json();
    console.log('Projections API Response:', data);  // Debug log
    return data;
};


// TRADE VALUE RANKING

export interface TradeValueRankings {
  real_id: number;
  name: string;
  team: string;
  position: string;
  contract_war: number;
  avg_war: number;
  total_contract: number;
  avg_contract: number;
  trade_value: number;
  control_through: number;
  years_control: number;
  total_future_war: number;
  total_future_value: number;
  historical_war: number;
  historical_value: number;
  contract_base_value: number;
}

export interface TradeValueRankingsResponse {
  players: TradeValueRankings[];
  total_count: number;
  total_pages: number;
  current_page: number;
}

export const getTradeValueRankings = async (
  params: {
    team?: string;
    page?: number;
    pageSize?: number;
    sortBy?: string;
    sortDirection?: 'asc' | 'desc';
  } = {}
): Promise<TradeValueRankingsResponse> => {
  const {
    team,
    page = 1,
    pageSize = 50,
    sortBy = 'trade_value',
    sortDirection = 'desc'
  } = params;

  const queryParams = new URLSearchParams({
    page: page.toString(),
    page_size: pageSize.toString(),
    sort_by: sortBy,
    sort_direction: sortDirection,
    ...(team && { team })
  });

  const response = await fetch(`${API_BASE}/trades/trade-val-rank?${queryParams}`);
  
  if (!response.ok) {
    throw new Error('Failed to fetch trade value rankings');
  }

  return response.json();
};