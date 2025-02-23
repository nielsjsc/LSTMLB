const getApiUrl = (): string => {
  if (import.meta.env.PROD) {
    return `${import.meta.env.VITE_API_URL}/api`;
  }
  return 'http://localhost:8000/api';
};

export const API_BASE: string = getApiUrl();
export const API_URL: string = API_BASE;