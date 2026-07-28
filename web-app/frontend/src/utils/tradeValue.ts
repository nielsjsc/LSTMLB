/**
 * Shared formatting and display helpers for trade-value components.
 *
 * Previously this logic was reimplemented separately in TeamPlayerList,
 * ValueDisplay, TradeMeter, and PlayerDetails, each with slightly
 * different behavior (rounding, sign handling, thresholds). Centralizing
 * it here means every component agrees on what "$0" and "balanced" mean.
 */

/** Formats a dollar value, e.g. 12_500_000 -> "$12.5M", -500_000 -> "-$0.5M" */
export const formatCurrency = (value: number | null | undefined): string => {
  if (value === null || value === undefined || isNaN(value)) return '—';
  const sign = value < 0 ? '-' : '';
  const abs = Math.abs(value);
  if (abs >= 1_000_000) return `${sign}$${(abs / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${sign}$${(abs / 1_000).toFixed(0)}K`;
  return `${sign}$${abs.toFixed(0)}`;
};

/**
 * Below this absolute surplus differential, a trade is treated as balanced
 * rather than favoring either side. Both TradeMeter and ValueDisplay's
 * summary footer read from this constant so they can't disagree with
 * each other (previously: 5,000,000 vs 2,000,000).
 */
export const BALANCED_TRADE_THRESHOLD = 2_000_000;

/**
 * Badge color classes for a position, tuned for contrast on a light
 * background (soft-tint bg + solid-weight text, e.g. bg-blue-50 /
 * text-blue-700) rather than the -400-on-dark-500/20 pairing, which
 * reads faint on white.
 *
 * Deliberately only two treatments, not one per position group: pitcher
 * vs. everyone else is the one distinction that's functionally meaningful
 * here (it's what determines war_bat vs. war_pit), so it's the only one
 * that gets color. Coloring every position group separately (infield,
 * outfield, DH...) added hues without adding signal.
 */
export const getPositionBadgeClasses = (position: string | undefined): string => {
  if (!position) return 'bg-gray-100 text-gray-600 border border-gray-200';
  const pos = position.toUpperCase();
  if (['SP', 'RP', 'CL', 'P'].includes(pos)) return 'bg-blue-50 text-blue-700 border border-blue-200';
  return 'bg-gray-100 text-gray-600 border border-gray-200';
};