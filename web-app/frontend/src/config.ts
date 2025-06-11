const getApiUrl = (): string => {
  if (import.meta.env.PROD) {
    // Remove the /api suffix since backend routes don't use it
    const apiUrl = import.meta.env.VITE_API_URL || 'https://43f4-71-212-206-128.ngrok-free.app';
    return apiUrl;  // Don't add /api
  }
  return 'http://localhost:8000';  // Don't add /api
};

export const API_BASE: string = getApiUrl();
export const API_URL: string = API_BASE;