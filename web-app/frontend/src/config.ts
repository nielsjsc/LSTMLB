const getApiUrl = (): string => {
  // In development, always use local backend
  if (import.meta.env.DEV) {
    return 'http://localhost:8000';
  }
  
  // In production, use environment variable (required)
  const apiUrl = import.meta.env.VITE_API_URL;
  if (!apiUrl) {
    console.error('VITE_API_URL not set in environment!');
    return '';
  }
  return apiUrl;
};

export const API_BASE: string = getApiUrl();
export const API_URL: string = API_BASE;

// Debug logging
console.log('Frontend Config Debug:');
console.log('import.meta.env.DEV:', import.meta.env.DEV);
console.log('import.meta.env.PROD:', import.meta.env.PROD);
console.log('Final API_BASE:', API_BASE);