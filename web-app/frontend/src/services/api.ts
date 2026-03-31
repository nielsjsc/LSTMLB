import { API_BASE as CONFIG_API_BASE, PROSPECT_DEFAULT_YEAR, CURRENT_YEAR } from '../config';

const API_BASE = CONFIG_API_BASE;

const createApiHeaders = () => {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    'ngrok-skip-browser-warning': 'true',
    'User-Agent': 'LongballAnalytics/1.0',
  };
  // Attach API key when configured (production)
  const apiKey = import.meta.env.VITE_API_KEY;
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }
  return headers;
};

// ── Request cancellation ────────────────────────────────────────────────
// Tracks one AbortController per keyed request group. Calling `cancellableFetch`
// with the same key aborts any in-flight request, preventing stale-response races.
const _controllers = new Map<string, AbortController>();

async function cancellableFetch(key: string, url: string, init: RequestInit = {}): Promise<Response> {
  _controllers.get(key)?.abort();
  const controller = new AbortController();
  _controllers.set(key, controller);
  try {
    const response = await fetch(url, { ...init, signal: controller.signal });
    return response;
  } finally {
    // Only clean up if this controller is still the active one
    if (_controllers.get(key) === controller) _controllers.delete(key);
  }
}


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
        mlb_id: number | null;
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
  year: number = PROSPECT_DEFAULT_YEAR
): Promise<Prospect[]> => {
  // Uses the unified /prospects/ endpoint with slim=true for minimal fields
  const params = new URLSearchParams({
    player_type: playerType,
    year: year.toString(),
    page_size: '500',
    slim: 'true'
  });
  const response = await fetch(`${API_BASE}/prospects?${params}`, {
    headers: createApiHeaders()
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(`Failed to fetch all prospects: ${JSON.stringify(errorData.detail)}`);
  }
  const data = await response.json();
  return data.players;
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
      headers: createApiHeaders(),  // Replace the existing headers
      body: JSON.stringify({
        team1_assets: team1Assets,
        team2_assets: team2Assets
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      const errorMessage = typeof errorData.detail === 'object' 
        ? JSON.stringify(errorData.detail) 
        : errorData.detail || 'Unknown error';
      throw new Error(`Failed to analyze trade: ${errorMessage}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    throw error;
  }
};


export const getPlayers = async (year: number = CURRENT_YEAR): Promise<Player[]> => {
    try {
        const response = await fetch(`${API_BASE}/players?year=${year}`, {
          headers: createApiHeaders()  // Add this line
        });
        if (!response.ok) throw new Error('Failed to fetch players');
        const data = await response.json();
        return data.players; // Return just the players array for Trade Analyzer
    } catch (error) {
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

  const response = await cancellableFetch('filterPlayers', url, {
    headers: createApiHeaders()
  });
  if (!response.ok) throw new Error('Failed to fetch players');
  return response.json();
};

export interface PlayerStats {
    name: string;
    team: string;
    position: string;
    mlb_id: number | null;
    isHistorical?: boolean;
    isProspectOnly?: boolean;
    headshot_url?: string;
    prospectData?: {
        prospect_id: number;
        IDfg: string | null;
        mlbam_id: number | null;
        name: string;
        org: string;
        position: string;
        age: number | null;
        fv: string;
        has_mlb: boolean;
        is_pitcher: boolean;
        tools: Record<string, string | null>;
        history: ProspectDetailHistory[];
    };
    historicalMeta?: {
        idfg: number;
        bbref: string | null;
        birth_year: number | null;
        death_year: number | null;
        first_year: number;
        last_year: number;
        teams: string[];
        career_war: number;
        career_bat_war: number;
        career_pit_war: number;
        career_salary: number | null;
        career_war_value: number;
        career_surplus: number | null;
    };
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
        salary?: number | null;
        war_value?: number | null;
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
            bat: number;
            bsr: number;
            def_value: number;
            hr: number;
            doubles: number;
            triples: number;
            r: number;
            rbi: number;
            sb: number;
            cs: number;
            xba?: number;
            xslg?: number;
            xwoba?: number;
        };
        pitching?: {
            g_pit: number;
            gs: number;
            war_pit: number;
            era: number;
            fip: number;
            k_pct_pit: number;
            bb_pct_pit: number;
            w?: number;
            l?: number;
            sv?: number;
            ip?: number;
            whip?: number;
            so?: number;
            bb?: number;
            k_9?: number;
            bb_9?: number;
            hr_9?: number;
            xera?: number;
        };
        
    }>;
    currentSeasonStats?: {
        batting?: {
            season: number;
            team: string;
            g: number | null;
            pa: number | null;
            ab: number | null;
            h: number | null;
            hr: number | null;
            doubles: number | null;
            triples: number | null;
            r: number | null;
            rbi: number | null;
            sb: number | null;
            cs: number | null;
            bb: number | null;
            so: number | null;
            avg: number | null;
            obp: number | null;
            slg: number | null;
            ops: number | null;
            woba: number | null;
            wrc_plus: number | null;
            bb_pct: number | null;
            k_pct: number | null;
            babip: number | null;
            war: number | null;
            bat: number | null;
            bsr: number | null;
            def_value: number | null;
        };
        pitching?: {
            season: number;
            team: string;
            g: number | null;
            gs: number | null;
            ip: number | null;
            w: number | null;
            l: number | null;
            sv: number | null;
            era: number | null;
            fip: number | null;
            k_pct: number | null;
            bb_pct: number | null;
            k_9: number | null;
            bb_9: number | null;
            hr_9: number | null;
            babip: number | null;
            whip: number | null;
            gb_pct: number | null;
            fb_pct: number | null;
            hr_fb: number | null;
            war: number | null;
        };
    };
}
  
export const getPlayerDetails = async (real_id: number): Promise<PlayerStats> => {
  const url = `${API_BASE}/players/${real_id}/details`;

  try {
      const response = await fetch(url, {
        headers: createApiHeaders()  // Add this line
      });
      if (!response.ok) {
          throw new Error('Failed to fetch player details');
      }
      const data = await response.json();
      return data;
  } catch (error) {
      throw error;
  }
};

export type BasePosition = 'C' | '1B' | '2B' | '3B' | 'SS' | 'OF' | 'LF' | 'CF' | 'RF' | 'DH' | 'SP' | 'RP';
export type PlayerPosition = string;
export interface ProjectionResponse {
    count: number;
    players: Array<{
        real_id: number;
        mlb_id: number | null;
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
            bat: number;
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
            ip: number;
            war_pit: number;
            era: number;
            fip: number;
            k_pct_pit: number;
            bb_pct_pit: number;
            gb_pct: number;
            fb_pct: number;
            hr_fb: number;
            hr_9: number;
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


  
export interface MiLBStatRow {
  season: number;
  team: string;
  level: string;
  age: number;
  // Hitter stats
  pa?: number;
  bb_pct?: number | null;
  k_pct?: number | null;
  avg?: number | null;
  obp?: number | null;
  slg?: number | null;
  ops?: number | null;
  iso?: number | null;
  babip?: number | null;
  woba?: number | null;
  wrc_plus?: number | null;
  spd?: number | null;
  // Pitcher stats
  ip?: number | null;
  k_9?: number | null;
  bb_9?: number | null;
  k_bb?: number | null;
  hr_9?: number | null;
  whip?: number | null;
  era?: number | null;
  fip?: number | null;
  xfip?: number | null;
}

export interface Prospect {
    id: number;
    playerId: string;
    IDfg: string | null;
    name: string;
    has_mlb: boolean;
    org: string;
    position: string;
    age: number | null;
    level: string | null;
    fv: string;
    value: number | null;
    composite: number | null;
    top_100: number | null;
    org_rank: number | null;

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

    // Stats (populated when view=stats or view=all_stats)
    latest_stats?: MiLBStatRow;
    all_stats?: MiLBStatRow[];
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
  year: number = PROSPECT_DEFAULT_YEAR,
  page: number = 1,
  pageSize: number = 25,
  sortBy?: string,
  sortDirection: 'asc' | 'desc' = 'asc',
  view: 'grades' | 'stats' | 'all_stats' = 'grades',
  minPa?: number,
  minIp?: number,
  minG?: number,
): Promise<ProspectResponse> => {
  const params = new URLSearchParams();
  params.append('player_type', playerType);
  params.append('year', year.toString());
  params.append('page', page.toString());
  params.append('page_size', pageSize.toString());
  params.append('view', view);
  if (team) params.append('team', team);
  if (position) params.append('position', position);
  if (sortBy) params.append('sort_by', sortBy);
  params.append('sort_direction', sortDirection);
  if (minPa && minPa > 0) params.append('min_pa', minPa.toString());
  if (minIp && minIp > 0) params.append('min_ip', minIp.toString());
  if (minG && minG > 0) params.append('min_g', minG.toString());
  
  const response = await cancellableFetch('getProspects', `${API_BASE}/prospects?${params}`, {
    headers: createApiHeaders()
  });
  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(`Failed to fetch prospects: ${errorData.detail || 'Unknown error'}`);
  }
  return response.json();
};

// ── Prospect Detail ──────────────────────────────────────────────────

export interface ProspectDetailHistory {
  year: number;
  age: number | null;
  org: string;
  position: string;
  fv: string;
  value: number | null;
  composite: number | null;
  top_100: number | null;
  org_rank: number | null;
  hit?: string;
  game_power?: string;
  raw_power?: string;
  speed?: string;
  fastball?: string;
  slider?: string;
  curve?: string;
  changeup?: string;
  command?: string;
}

export interface ProspectDetail {
  id: number;
  IDfg: number | null;
  mlbam_id: number | null;
  name: string;
  org: string;
  position: string;
  age: number | null;
  fv: string;
  has_mlb: boolean;
  is_pitcher: boolean;
  tools: Record<string, string | null>;
  history: ProspectDetailHistory[];
  mlb_info: {
    mlbam_id?: number;
    mlb_id?: number;
    headshot_url: string;
  } | null;
}

export const getProspectDetail = async (prospectId: number): Promise<ProspectDetail> => {
  const url = `${API_BASE}/prospects/${prospectId}`;
  const response = await fetch(url, { headers: createApiHeaders() });
  if (!response.ok) {
    throw new Error(`Failed to fetch prospect detail: ${response.status}`);
  }
  return response.json();
};

// ── MiLB Stats ────────────────────────────────────────────────────────

export interface MiLBHittingSeason {
  season: number;
  team: string;
  level: string;
  age: number;
  pa: number;
  bb_pct: number | null;
  k_pct: number | null;
  avg: number | null;
  obp: number | null;
  slg: number | null;
  ops: number | null;
  iso: number | null;
  babip: number | null;
  woba: number | null;
  wrc_plus: number | null;
  spd: number | null;
}

export interface MiLBPitchingSeason {
  season: number;
  team: string;
  level: string;
  age: number;
  ip: number | null;
  k_9: number | null;
  bb_9: number | null;
  k_bb: number | null;
  hr_9: number | null;
  k_pct: number | null;
  bb_pct: number | null;
  avg: number | null;
  whip: number | null;
  babip: number | null;
  era: number | null;
  fip: number | null;
  xfip: number | null;
}

export interface MiLBStatsResponse {
  hitting: MiLBHittingSeason[];
  pitching: MiLBPitchingSeason[];
}

export const getProspectMiLBStats = async (prospectId: number): Promise<MiLBStatsResponse> => {
  const url = `${API_BASE}/prospects/${prospectId}/milb-stats`;
  const response = await fetch(url, { headers: createApiHeaders() });
  if (!response.ok) {
    return { hitting: [], pitching: [] };
  }
  return response.json();
};

export const getPlayerMiLBStats = async (playerId: number): Promise<MiLBStatsResponse> => {
  const url = `${API_BASE}/players/${playerId}/milb-stats`;
  const response = await fetch(url, { headers: createApiHeaders() });
  if (!response.ok) {
    return { hitting: [], pitching: [] };
  }
  return response.json();
};

  export const getProjections = async (
  year: number,
  playerType: 'hitter' | 'pitcher',
  team?: string,
  position?: string,
  page: number = 1,
  pageSize: number = 50,
  sortBy?: string,
  sortDirection: 'asc' | 'desc' = 'desc',
  projectionType: 'ros' | 'preseason' = 'ros'
): Promise<ProjectionResponse> => {
  const params = new URLSearchParams({
      year: year.toString(),
      player_type: playerType,
      page: page.toString(),
      page_size: pageSize.toString(),
      projection_type: projectionType,
  });

  if (team) params.append('team', team);
  if (position) params.append('position', position);
  if (sortBy) params.append('sort_by', sortBy);
  if (sortDirection) params.append('sort_direction', sortDirection);

  const response = await cancellableFetch('getProjections', `${API_BASE}/projections?${params}`, {
    headers: createApiHeaders()
  });
  if (!response.ok) throw new Error('Failed to fetch projections');
  return response.json();
};

// TRADE VALUE RANKING

export interface TradeValueRankings {
  real_id: number;
  mlb_id: number | null;
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

  const url = `${API_BASE}/trades/trade-val-rank?${queryParams}`;

  const response = await cancellableFetch('getTradeValueRankings', url, {
    headers: createApiHeaders()
  });
  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Failed to fetch trade value rankings: ${errorText}`);
  }
  return response.json();
};

// ── Trade Value History (per-player timeline) ─────────────────────────────
export interface TradeValuePoint {
  year: number;
  date: string | null;              // YYYY-MM-DD
  value: number;
  valueType: 'prospect' | 'mlb_surplus' | 'free_agent';
  transactionType: string | null;   // Spotrac txn type (traded, fa_signing, …)
  label: string;
  yearsControl: number | null;
  projectedWar: number | null;
  projectedSalary: number | null;
  warPerYear: number | null;
}

export const getTradeValueHistory = async (
  playerId: number,
  opts?: { granularity?: string; startDate?: string; endDate?: string },
): Promise<TradeValuePoint[]> => {
  const params = new URLSearchParams();
  if (opts?.granularity) params.set('granularity', opts.granularity);
  if (opts?.startDate) params.set('start_date', opts.startDate);
  if (opts?.endDate) params.set('end_date', opts.endDate);
  const qs = params.toString();
  const url = `${API_BASE}/players/${playerId}/trade-value-history${qs ? `?${qs}` : ''}`;
  try {
    const response = await fetch(url, { headers: createApiHeaders() });
    if (!response.ok) return [];
    return await response.json();
  } catch {
    return [];
  }
};

// ── Fielding Stats ────────────────────────────────────────────────────────
export interface FieldingStat {
  season: number;
  team: string | null;
  pos: string;
  age: number | null;
  g: number | null;
  gs: number | null;
  inn: number | null;
  sc_total_runs: number | null;
  sc_range_runs: number | null;
  sc_arm_runs: number | null;
  sc_dp_runs: number | null;
  sc_framing_runs: number | null;
  sc_throwing_runs: number | null;
  sc_blocking_runs: number | null;
  drs: number | null;
  uzr: number | null;
  uzr_150: number | null;
  oaa: number | null;
  errors: number | null;
  fp: number | null;
  is_projection: boolean;
}

export const getPlayerFieldingStats = async (playerId: number): Promise<FieldingStat[]> => {
  const url = `${API_BASE}/players/${playerId}/fielding-stats`;
  try {
    const response = await fetch(url, { headers: createApiHeaders() });
    if (!response.ok) return [];
    return await response.json();
  } catch {
    return [];
  }
};

// ── Player Bio / Awards / Draft ───────────────────────────────────────────
export interface PlayerAward {
  name: string;
  season: string;
}

export interface PlayerDraft {
  year: string;
  round: string;
  pickNumber: number;
  school: string | null;
  team: string | null;
}

export interface PlayerBio {
  height: string | null;
  weight: number | null;
  birthDate: string | null;
  birthCity: string | null;
  birthStateProvince: string | null;
  birthCountry: string | null;
  batSide: string | null;
  pitchHand: string | null;
  mlbDebutDate: string | null;
  primaryNumber: string | null;
  nickName: string | null;
}

export interface PlayerInfo {
  bio: PlayerBio;
  awards: PlayerAward[];
  draft: PlayerDraft | null;
}

export const getPlayerInfo = async (playerId: number): Promise<PlayerInfo | null> => {
  const url = `${API_BASE}/players/${playerId}/info`;
  try {
    const response = await fetch(url, { headers: createApiHeaders() });
    if (!response.ok) return null;
    const data = await response.json();
    if (!data.bio) return null;
    return data;
  } catch {
    return null;
  }
};

// ── Transaction History ───────────────────────────────────────────────────
export interface LinkedPlayer {
  name: string;
  mlbId: number;
  realId: number | null;
}

export interface Transaction {
  id: number | string;
  date: string;
  typeCode: string;
  typeDesc: string;
  description: string;
  fromTeam: string;
  fromTeamName: string;
  toTeam: string;
  toTeamName: string;
  linkedPlayers: LinkedPlayer[];
}

export const getPlayerTransactions = async (playerId: number): Promise<Transaction[]> => {
  const url = `${API_BASE}/players/${playerId}/transactions`;
  try {
    const response = await fetch(url, { headers: createApiHeaders() });
    if (!response.ok) return [];
    return await response.json();
  } catch {
    return [];
  }
};

// ═══════════════════════════════════════════════════════════════════════════════
//  PAST TRADES
// ═══════════════════════════════════════════════════════════════════════════════

export interface TradePlayerSummary {
  mlb_id: number;
  name: string;
  war_with_team: number;
  surplus: number;
  prospect_fv: number | null;
  from_team: string;
  from_team_name: string;
  // Prospect linking
  prospect_id?: number | null;
  has_data?: boolean;
  // Projected fields (present for "projected" evaluation_type trades)
  projected_war?: number | null;
  projected_surplus?: number | null;
  has_projection?: boolean;
}

export interface TradePlayerDetail extends TradePlayerSummary {
  to_team: string;
  to_team_name: string;
  seasons_with_team: number;
  yearly_war: { year: number; war: number }[];
  salary_with_team: number;
  war_value: number;
  still_on_team: boolean;
  departure_year: number | null;
  prospect_rank: number | null;
  prospect_top_100: boolean | null;
  prospect_level: string | null;
  prospect_value?: number | null;
  // Projected fields (present for "projected" evaluation_type trades)
  projected_war_value?: number | null;
  projected_salary?: number | null;
  projected_yearly_war?: { year: number; war: number }[];
  contract_remaining?: number | null;
}

export interface TradeSideSummary {
  team: string;
  team_name: string;
  total_war: number;
  total_salary: number;
  total_war_value: number;
  total_surplus: number;
  players_received: TradePlayerSummary[];
  // Projected side totals (present for "projected" trades)
  projected_total_war?: number;
  projected_total_surplus?: number;
}

export interface TradeSideDetail {
  team: string;
  team_name: string;
  total_war: number;
  total_salary: number;
  total_war_value: number;
  total_surplus: number;
  players_received: TradePlayerDetail[];
  // Projected side totals (present for "projected" trades)
  projected_total_war?: number;
  projected_total_war_value?: number;
  projected_total_salary?: number;
  projected_total_surplus?: number;
}

export type EvaluationType = 'actual' | 'projected';
export type EvaluationConfidence = 'definitive' | 'maturing' | 'early' | 'projected';

export interface PastTradeSummary {
  trade_id: number;
  date: string;
  year: number;
  description: string;
  has_cash: boolean;
  has_ptbnl: boolean;
  n_teams: number;
  n_players: number;
  winner: string;
  winner_name: string;
  loser: string;
  loser_name: string;
  surplus_diff: number;
  total_trade_war: number;
  max_prospect_fv: number | null;
  evaluation_type: EvaluationType;
  evaluation_confidence: EvaluationConfidence;
  is_featured: boolean;
  projected_total_war?: number;
  projected_surplus_diff?: number;
  sides: TradeSideSummary[];
}

export interface PastTradeDetail {
  trade_id: number;
  date: string;
  year: number;
  description: string;
  has_cash: boolean;
  has_ptbnl: boolean;
  n_teams: number;
  n_players: number;
  winner: string;
  winner_name: string;
  loser: string;
  loser_name: string;
  surplus_diff: number;
  total_trade_war: number;
  max_prospect_fv: number | null;
  evaluation_type: EvaluationType;
  evaluation_confidence: EvaluationConfidence;
  is_featured: boolean;
  projected_total_war?: number;
  projected_surplus_diff?: number;
  sides: TradeSideDetail[];
}

export interface PastTradesResponse {
  trades: PastTradeSummary[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export const getPastTrades = async (params?: {
  page?: number;
  page_size?: number;
  sort_by?: string;
  sort_dir?: string;
  team?: string;
  year?: number;
  min_war?: number;
  search?: string;
  featured?: boolean;
  confidence?: string;
}): Promise<PastTradesResponse> => {
  const searchParams = new URLSearchParams();
  if (params?.page) searchParams.set('page', String(params.page));
  if (params?.page_size) searchParams.set('page_size', String(params.page_size));
  if (params?.sort_by) searchParams.set('sort_by', params.sort_by);
  if (params?.sort_dir) searchParams.set('sort_dir', params.sort_dir);
  if (params?.team) searchParams.set('team', params.team);
  if (params?.year) searchParams.set('year', String(params.year));
  if (params?.min_war) searchParams.set('min_war', String(params.min_war));
  if (params?.search) searchParams.set('search', params.search);
  if (params?.featured !== undefined) searchParams.set('featured', String(params.featured));
  if (params?.confidence) searchParams.set('confidence', params.confidence);

  const url = `${API_BASE}/trades/past-trades?${searchParams.toString()}`;
  const response = await cancellableFetch('getPastTrades', url, { headers: createApiHeaders() });
  if (!response.ok) throw new Error('Failed to fetch past trades');
  return response.json();
};

export const getPastTradeDetail = async (tradeId: number): Promise<PastTradeDetail> => {
  const url = `${API_BASE}/trades/past-trades/${tradeId}`;
  const response = await fetch(url, { headers: createApiHeaders() });
  if (!response.ok) throw new Error('Trade not found');
  return response.json();
};

export const getPlayerPastTrades = async (mlbId: number): Promise<{ trades: PastTradeDetail[] }> => {
  const url = `${API_BASE}/trades/player-trades/${mlbId}`;
  const response = await fetch(url, { headers: createApiHeaders() });
  if (!response.ok) return { trades: [] };
  return response.json();
};

// ── Historical Player Search ──────────────────────────────────────────

export interface HistoricalPlayerSummary {
  idfg: number;
  mlbam: number | null;
  name: string;
  teams: string[];
  first_year: number;
  last_year: number;
  career_war: number;
  is_pitcher: boolean;
}

export interface HistoricalSearchResponse {
  total: number;
  offset: number;
  limit: number;
  players: HistoricalPlayerSummary[];
}

export const searchHistoricalPlayers = async (params?: {
  q?: string;
  team?: string;
  min_war?: number;
  decade?: number;
  limit?: number;
  offset?: number;
}): Promise<HistoricalSearchResponse> => {
  const searchParams = new URLSearchParams();
  if (params?.q) searchParams.set('q', params.q);
  if (params?.team) searchParams.set('team', params.team);
  if (params?.min_war) searchParams.set('min_war', String(params.min_war));
  if (params?.decade) searchParams.set('decade', String(params.decade));
  if (params?.limit) searchParams.set('limit', String(params.limit));
  if (params?.offset) searchParams.set('offset', String(params.offset));

  const url = `${API_BASE}/historical/search?${searchParams.toString()}`;
  const response = await cancellableFetch('searchHistoricalPlayers', url, { headers: createApiHeaders() });
  if (!response.ok) throw new Error('Failed to search historical players');
  return response.json();
};