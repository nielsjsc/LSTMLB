import { QueryClient } from '@tanstack/react-query';

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000,   // 5 minutes — projection data is stable within a session
      gcTime: 10 * 60 * 1000,     // 10 minutes — keep cached data after unmount
      refetchOnWindowFocus: false, // no surprise refetches on tab switch
      retry: 1,                    // single retry on transient failures
    },
  },
});
