const getApiUrl = (): string => {
  // In development, always use local backend
  if (import.meta.env.DEV) {
    return 'http://localhost:8000';
  }
  
  // In production, use environment variable with fallback to current ngrok
  const apiUrl = import.meta.env.VITE_API_URL || 'https://80940e817e17.ngrok-free.app';
  if (!import.meta.env.VITE_API_URL) {
    console.warn('VITE_API_URL not set - using hardcoded ngrok URL');
  }
  return apiUrl;
};

export const API_BASE: string = getApiUrl();
export const API_URL: string = API_BASE;

// ── Season configuration ────────────────────────────────────────────────
// Update these once per season — every downstream reference derives from here.

// Current projection year
export const CURRENT_YEAR = 2026;

// Number of future years to display in projections (e.g., 5 means show 2026-2030)
export const MAX_PROJECTION_YEARS = 5;

// Prospect data availability (oldest → newest)
export const PROSPECT_YEAR_START = 2014;
export const PROSPECT_YEAR_END = 2025;
export const PROSPECT_YEARS: number[] = Array.from(
  { length: PROSPECT_YEAR_END - PROSPECT_YEAR_START + 1 },
  (_, i) => PROSPECT_YEAR_END - i  // newest first for dropdown
);
export const PROSPECT_DEFAULT_YEAR = 2025;  // 2025 has full boards (900 prospects)