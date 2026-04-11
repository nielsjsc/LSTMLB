import { useQuery, keepPreviousData } from '@tanstack/react-query';
import { useMemo } from 'react';
import {
  getProjections,
  getProspects,
  getTradeValueRankings,
  getPastTrades,
  getPlayers,
  getAllProspects,
} from '../services/api';
import { CURRENT_YEAR, PROSPECT_DEFAULT_YEAR } from '../config';

// ── Projections ──────────────────────────────────────────────────────────────

export function useProjections(params: {
  year: number;
  playerType: 'hitter' | 'pitcher';
  team?: string;
  position?: string;
  projectionType?: 'ros' | 'preseason';
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
}) {
  return useQuery({
    queryKey: ['projections', params] as const,
    queryFn: () =>
      getProjections(
        params.year,
        params.playerType,
        params.team,
        params.position,
        params.page ?? 1,
        params.pageSize ?? 50,
        params.sortBy,
        params.sortDirection ?? 'desc',
        params.projectionType ?? 'ros',
      ),
    placeholderData: keepPreviousData,
  });
}

// ── Prospects ────────────────────────────────────────────────────────────────

export function useProspects(params: {
  playerType: 'hitter' | 'pitcher';
  team?: string;
  position?: string;
  year?: number;
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
  view?: 'grades' | 'stats' | 'all_stats';
  minPa?: number;
  minIp?: number;
  minG?: number;
}) {
  return useQuery({
    queryKey: ['prospects', params] as const,
    queryFn: () =>
      getProspects(
        params.playerType,
        params.team,
        params.position,
        params.year ?? PROSPECT_DEFAULT_YEAR,
        params.page ?? 1,
        params.pageSize ?? 50,
        params.sortBy,
        params.sortDirection ?? 'asc',
        params.view ?? 'grades',
        params.minPa,
        params.minIp,
        params.minG,
      ),
    placeholderData: keepPreviousData,
  });
}

// ── Trade Value Rankings ─────────────────────────────────────────────────────

export function useTradeValues(params: {
  team?: string;
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortDirection?: 'asc' | 'desc';
}) {
  return useQuery({
    queryKey: ['tradeValues', params] as const,
    queryFn: () =>
      getTradeValueRankings({
        team: params.team,
        page: params.page ?? 1,
        pageSize: params.pageSize ?? 50,
        sortBy: params.sortBy ?? 'trade_value',
        sortDirection: params.sortDirection ?? 'desc',
      }),
    placeholderData: keepPreviousData,
  });
}

// ── Past Trades ──────────────────────────────────────────────────────────────

export function usePastTrades(params: {
  page?: number;
  pageSize?: number;
  sortBy?: string;
  sortDir?: string;
  team?: string;
  year?: number | '';
  search?: string;
  featured?: boolean;
  confidence?: string;
  minWar?: number;
}) {
  return useQuery({
    queryKey: ['pastTrades', params] as const,
    queryFn: () =>
      getPastTrades({
        page: params.page ?? 1,
        page_size: params.pageSize ?? 25,
        sort_by: params.sortBy,
        sort_dir: params.sortDir,
        team: params.team || undefined,
        year: params.year || undefined,
        search: params.search || undefined,
        featured: params.featured,
        confidence: params.confidence || undefined,
        min_war: params.minWar,
      }),
    placeholderData: keepPreviousData,
  });
}

// ── Trade Simulator Assets ───────────────────────────────────────────────────

export function useTradeAssets() {
  const playersQuery = useQuery({
    queryKey: ['players', CURRENT_YEAR] as const,
    queryFn: () => getPlayers(CURRENT_YEAR),
    staleTime: 15 * 60 * 1000, // 15 min — large, rarely-changing dataset
  });

  const hittersQuery = useQuery({
    queryKey: ['prospects', 'all', 'hitter', PROSPECT_DEFAULT_YEAR] as const,
    queryFn: () => getAllProspects('hitter', PROSPECT_DEFAULT_YEAR),
    staleTime: 15 * 60 * 1000,
  });

  const pitchersQuery = useQuery({
    queryKey: ['prospects', 'all', 'pitcher', PROSPECT_DEFAULT_YEAR] as const,
    queryFn: () => getAllProspects('pitcher', PROSPECT_DEFAULT_YEAR),
    staleTime: 15 * 60 * 1000,
  });

  const prospects = useMemo(() => {
    if (!hittersQuery.data || !pitchersQuery.data) return [];
    return [...hittersQuery.data, ...pitchersQuery.data].sort(
      (a, b) => (b.value || 0) - (a.value || 0),
    );
  }, [hittersQuery.data, pitchersQuery.data]);

  return {
    players: playersQuery.data ?? [],
    prospects,
    isLoading:
      playersQuery.isLoading || hittersQuery.isLoading || pitchersQuery.isLoading,
    error:
      playersQuery.error || hittersQuery.error || pitchersQuery.error,
  };
}
