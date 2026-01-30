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

// Current projection year
export const CURRENT_YEAR = 2026;

// Debug logging
console.log('Frontend Config Debug:');
console.log('import.meta.env.DEV:', import.meta.env.DEV);
console.log('import.meta.env.PROD:', import.meta.env.PROD);
console.log('Final API_BASE:', API_BASE);