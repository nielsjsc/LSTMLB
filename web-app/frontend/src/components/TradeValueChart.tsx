import React, { useState, useRef, useEffect, useMemo } from 'react';
import type { TradeValuePoint, Transaction } from '../services/api';

// ─── Helpers ────────────────────────────────────────────────
const fmtDollar = (v: number): string => {
  const abs = Math.abs(v);
  if (abs >= 1_000_000_000) return `${v < 0 ? '-' : ''}$${(abs / 1_000_000_000).toFixed(1)}B`;
  if (abs >= 1_000_000) return `${v < 0 ? '-' : ''}$${(abs / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${v < 0 ? '-' : ''}$${(abs / 1_000).toFixed(0)}K`;
  return `$${v.toFixed(0)}`;
};

const fmtDollarFull = (v: number): string => {
  const abs = Math.abs(v);
  if (abs >= 1_000_000_000) return `${v < 0 ? '-' : ''}$${(abs / 1_000_000_000).toFixed(2)}B`;
  if (abs >= 1_000_000) return `${v < 0 ? '-' : ''}$${(abs / 1_000_000).toFixed(1)}M`;
  if (abs >= 1_000) return `${v < 0 ? '-' : ''}$${(abs / 1_000).toFixed(0)}K`;
  return `$${v.toFixed(0)}`;
};

const fmtWar = (v: number): string => v.toFixed(1);

// Human-readable transaction-type label
const txnLabel = (t: string | null): string => {
  if (!t) return '';
  const map: Record<string, string> = {
    fa_signing: 'FA Signing',
    traded: 'Traded',
    extension: 'Extension',
    elected_fa: 'Free Agent',
    released: 'Released',
    dfa: 'DFA',
    claimed: 'Claimed',
    drafted: 'Drafted',
    international_signing: 'International Signing',
    option: 'Optioned',
    recall: 'Recalled',
    other: 'Other',
  };
  return map[t] ?? t.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
};

// Convert date string (YYYY-MM-DD) to fractional year for X-axis positioning
const dateToFx = (date: string | null | undefined, year: number): number => {
  if (!date) return year + 0.16; // ~March 1 default
  const d = new Date(date + 'T12:00:00');
  if (isNaN(d.getTime())) return year + 0.16;
  const yr = d.getFullYear();
  const start = new Date(yr, 0, 1).getTime();
  const end = new Date(yr + 1, 0, 1).getTime();
  return yr + (d.getTime() - start) / (end - start);
};

/**
 * Adaptive power-curve scale for the Y axis.
 * Picks an exponent based on each player's data range:
 *   - If values span many orders of magnitude (e.g. $1M → $400M), use strong
 *     compression (low exponent) so small values are still visible.
 *   - If values are clustered (e.g. $300M → $500M), stay near-linear so
 *     differences within that band are clearly visible.
 *
 * powScale(v, exp) = sign(v) * |v|^exp   (exp in 0..1, 1 = fully linear)
 */
const powScale = (v: number, exp: number): number =>
  Math.sign(v) * Math.pow(Math.abs(v), exp);

/** Derive the best exponent for a given value array.
 *  Ratio = min(abs) / max(abs).  High ratio → clustered → linear.
 *  Low ratio → wide spread → compress. */
const deriveExponent = (values: number[]): number => {
  const absVals = values.map((v) => Math.abs(v)).filter((v) => v > 0);
  if (absVals.length < 2) return 1; // linear if only one point
  const lo = Math.min(...absVals);
  const hi = Math.max(...absVals);
  if (hi === 0) return 1;
  const ratio = lo / hi; // 0..1  (1 = all same magnitude)
  // Map ratio → exponent:  ratio=1 → exp=1 (linear), ratio→0 → exp≈0.35
  return 0.35 + 0.65 * ratio;
};

// ─── Chart Dimensions ───────────────────────────────────────
const MARGIN_DEFAULT = { top: 20, right: 20, bottom: 36, left: 60 };
const MARGIN_MOBILE  = { top: 16, right: 12, bottom: 32, left: 44 };
const MOBILE_BREAKPOINT = 480;
const DOT_RADIUS = 5;
const HOVER_RADIUS = 7;

// ─── Dot color by value type ────────────────────────────────
const DOT_COLOR: Record<string, string> = {
  prospect: '#a78bfa',      // violet-400
  mlb_surplus: '#d97706',   // amber-600 (brand)
  free_agent: '#e11d48',    // rose-600
};
const DEFAULT_DOT_COLOR = '#6b7280'; // gray-500

// ─── Component ──────────────────────────────────────────────
interface Props {
  data: TradeValuePoint[];
  teamColor: string;
  teamAccent: string;
  transactions?: Transaction[];
}

// Transaction types worth annotating on the chart
const IMPORTANT_TXN = new Set(['TR', 'SGN', 'SFA', 'FA', 'EXT', 'DFA']);

const TradeValueChart: React.FC<Props> = ({ data, teamColor, teamAccent, transactions }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [pinnedIdx, setPinnedIdx] = useState<number | null>(null);
  const activeIdx = pinnedIdx ?? hoverIdx;
  const [dims, setDims] = useState({ width: 700, height: 300 });
  const isMobile = dims.width < MOBILE_BREAKPOINT;
  const MARGIN = isMobile ? MARGIN_MOBILE : MARGIN_DEFAULT;

  // Responsive width via ResizeObserver
  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        if (w > 0) {
          // On mobile, use a taller aspect ratio so the chart isn't so squished
          const minH = w < MOBILE_BREAKPOINT ? 240 : 220;
          const ratio = w < MOBILE_BREAKPOINT ? 0.55 : 0.4;
          setDims({ width: w, height: Math.min(360, Math.max(minH, w * ratio)) });
        }
      }
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  // Sort by date (fractional year) for proper within-year positioning
  const sorted = useMemo(
    () =>
      [...data]
        .map((d) => ({ ...d, _fx: dateToFx(d.date, d.year) }))
        .sort((a, b) => a._fx - b._fx),
    [data],
  );

  // Detect in-season year: the year with ≥ 3 data points (weekly/daily data)
  const inSeasonYear = useMemo(() => {
    const counts: Record<number, number> = {};
    sorted.forEach((d) => { counts[d.year] = (counts[d.year] || 0) + 1; });
    let best = 0, bestCount = 0;
    for (const [yr, c] of Object.entries(counts)) {
      if (c > bestCount) { bestCount = c; best = Number(yr); }
    }
    return bestCount >= 3 ? best : null;
  }, [sorted]);

  // Keyboard navigation when a card is pinned
  useEffect(() => {
    if (pinnedIdx == null) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
        e.preventDefault();
        setPinnedIdx((p) => (p != null && p > 0 ? p - 1 : p));
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
        e.preventDefault();
        setPinnedIdx((p) => (p != null && p < sorted.length - 1 ? p + 1 : p));
      } else if (e.key === 'Escape') {
        setPinnedIdx(null);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [pinnedIdx, sorted.length]);

  if (sorted.length === 0) return null;

  const plotW = dims.width - MARGIN.left - MARGIN.right;
  const plotH = dims.height - MARGIN.top - MARGIN.bottom;

  // ── Scales ──────────────────────────────────────────
  const fxValues = sorted.map((d) => d._fx);
  const minFx = fxValues[0];
  const maxFx = fxValues[fxValues.length - 1];
  const fxSpan = Math.max(maxFx - minFx, 1);

  // Unique integer years for x-axis labels
  const uniqueYears = useMemo(() => {
    const seen = new Set<number>();
    return sorted
      .map((d) => d.year)
      .filter((yr) => {
        if (seen.has(yr)) return false;
        seen.add(yr);
        return true;
      });
  }, [sorted]);

  const values = sorted.map((d) => d.value);
  const rawMax = Math.max(...values);
  const rawMin = Math.min(0, ...values);
  const valRange = rawMax - rawMin || 1;
  const yPad = valRange * 0.12;
  const yMax = rawMax + yPad;
  const yMin = rawMin - (rawMin < 0 ? yPad : 0);
  const yRange = yMax - yMin || 1;

  // ── X scale: log-compressed time (recent dates get more space) ──
  const logFxMax = Math.log(1 + fxSpan);
  const x = (fx: number) => {
    if (fxSpan === 0) return MARGIN.left + plotW / 2;
    const dist = maxFx - fx;
    return MARGIN.left + (1 - Math.log(1 + dist) / logFxMax) * plotW;
  };

  // ── Y scale: adaptive power curve (per-player optimal compression) ──
  const yExp = useMemo(() => deriveExponent(values), [values]);
  const sYMin = powScale(yMin, yExp);
  const sYMax = powScale(yMax, yExp);
  const sYRange = sYMax - sYMin || 1;
  const y = (val: number) => MARGIN.top + (1 - (powScale(val, yExp) - sYMin) / sYRange) * plotH;

  // ── Smooth curve helpers (Catmull-Rom → cubic bezier) ─────────────────
  const smoothPath = (pts: { x: number; y: number }[]): string => {
    if (pts.length < 2) return '';
    if (pts.length === 2) return `M${pts[0].x},${pts[0].y}L${pts[1].x},${pts[1].y}`;
    let d = `M${pts[0].x},${pts[0].y}`;
    for (let i = 0; i < pts.length - 1; i++) {
      const p0 = pts[Math.max(i - 1, 0)];
      const p1 = pts[i];
      const p2 = pts[i + 1];
      const p3 = pts[Math.min(i + 2, pts.length - 1)];
      const cp1x = p1.x + (p2.x - p0.x) / 6;
      const cp1y = p1.y + (p2.y - p0.y) / 6;
      const cp2x = p2.x - (p3.x - p1.x) / 6;
      const cp2y = p2.y - (p3.y - p1.y) / 6;
      d += ` C${cp1x},${cp1y} ${cp2x},${cp2y} ${p2.x},${p2.y}`;
    }
    return d;
  };

  // ── Path (line + area) — connect ALL data points with smooth curve ──
  const linePoints = sorted.map((d) => ({ x: x(d._fx), y: y(d.value) }));
  const linePath = smoothPath(linePoints);
  const areaPath =
    linePoints.length > 1
      ? `${linePath}L${linePoints[linePoints.length - 1].x},${y(0)}L${linePoints[0].x},${y(0)}Z`
      : '';

  // ── Transaction annotations (positioned on the curve) ──────────────
  const chartTxns = useMemo(() => {
    if (!transactions?.length) return [];
    return transactions
      .filter((t) => IMPORTANT_TXN.has(t.typeCode) && t.date)
      .map((t) => {
        const d = new Date(t.date + 'T12:00:00');
        const yr = isNaN(d.getTime()) ? 0 : d.getFullYear();
        return { ...t, _fx: dateToFx(t.date, yr) };
      })
      .filter((t) => t._fx >= minFx && t._fx <= maxFx);
  }, [transactions, minFx, maxFx]);

  // Linear-interpolate trade value at an arbitrary fractional-year position
  const interpolateValue = (fx: number): number => {
    if (sorted.length === 0) return 0;
    if (sorted.length === 1) return sorted[0].value;
    if (fx <= sorted[0]._fx) return sorted[0].value;
    if (fx >= sorted[sorted.length - 1]._fx) return sorted[sorted.length - 1].value;
    for (let i = 0; i < sorted.length - 1; i++) {
      if (sorted[i]._fx <= fx && sorted[i + 1]._fx >= fx) {
        const t = (fx - sorted[i]._fx) / (sorted[i + 1]._fx - sorted[i]._fx);
        return sorted[i].value + t * (sorted[i + 1].value - sorted[i].value);
      }
    }
    return sorted[sorted.length - 1].value;
  };

  // ── Gridlines ───────────────────────────────────────
  // ── Y-axis ticks (adaptive to data range) ──
  const yTicks = useMemo(() => {
    const range = yMax - yMin;
    if (range === 0) return [0];
    // Pick a "nice" step based on the data range
    const rawStep = range / 5;
    const mag = Math.pow(10, Math.floor(Math.log10(rawStep)));
    const nice = [1, 2, 2.5, 5, 10];
    const niceStep = mag * (nice.find((n) => n * mag >= rawStep) ?? 10);
    const ticks: number[] = [];
    let t = Math.ceil(yMin / niceStep) * niceStep;
    while (t <= yMax + niceStep * 0.01) {
      ticks.push(t);
      t += niceStep;
    }
    return ticks.length > 0 ? ticks : [0];
  }, [yMin, yMax]);

  // Build tooltip data
  const hovered = activeIdx != null ? sorted[activeIdx] : null;
  const canPrev = activeIdx != null && activeIdx > 0;
  const canNext = activeIdx != null && activeIdx < sorted.length - 1;
  const goPrev = () => { if (canPrev) { setPinnedIdx(activeIdx! - 1); } };
  const goNext = () => { if (canNext) { setPinnedIdx(activeIdx! + 1); } };

  return (
    <div ref={containerRef} className="w-full relative">
      <svg
        ref={svgRef}
        width={dims.width}
        height={dims.height}
        className="select-none"
        onMouseLeave={() => { if (pinnedIdx == null) setHoverIdx(null); }}
      >
        <defs>
          <linearGradient id="tvAreaGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={teamColor} stopOpacity={0.25} />
            <stop offset="100%" stopColor={teamColor} stopOpacity={0.02} />
          </linearGradient>
        </defs>

        {/* ── Y gridlines ─────────────────────── */}
        {yTicks.map((t) => (
          <g key={t}>
            <line
              x1={MARGIN.left}
              x2={dims.width - MARGIN.right}
              y1={y(t)}
              y2={y(t)}
              stroke="rgba(0,0,0,0.06)"
              strokeWidth={1}
            />
            <text
              x={MARGIN.left - 6}
              y={y(t)}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-gray-500"
              fontSize={isMobile ? 9 : 11}
              fontFamily="inherit"
            >
              {fmtDollar(t)}
            </text>
          </g>
        ))}

        {/* ── Zero line ──────────────────────── */}
        {yMin < 0 && (
          <line
            x1={MARGIN.left}
            x2={dims.width - MARGIN.right}
            y1={y(0)}
            y2={y(0)}
            stroke="rgba(0,0,0,0.12)"
            strokeWidth={1}
            strokeDasharray="4,3"
          />
        )}

        {/* ── Area fill ───────────────────────── */}
        <path d={areaPath} fill="url(#tvAreaGrad)" />

        {/* ── Main line ───────────────────────── */}
        <path
          d={linePath}
          fill="none"
          stroke={teamAccent}
          strokeWidth={2.5}
          strokeLinejoin="round"
          strokeLinecap="round"
        />

        {/* ── Data dots — historical years get full dots; in-season only on hover ── */}
        {sorted.map((d, i) => {
          const isInSeason = d.year === inSeasonYear;
          const color = DOT_COLOR[d.valueType] ?? DEFAULT_DOT_COLOR;
          const isHovered = activeIdx === i;
          // In-season points: hide dot unless hovered
          if (isInSeason && !isHovered) return null;
          const dotCx = x(d._fx);
          const dotCy = y(d.value);
          const baseR = isMobile ? 3.5 : DOT_RADIUS;
          const hoverR = isMobile ? 5.5 : HOVER_RADIUS;
          const r = isHovered ? hoverR : baseR;
          return (
            <g key={`dot-${i}`}>
              {isHovered && (
                <circle cx={dotCx} cy={dotCy} r={hoverR + 4} fill={color} opacity={0.18} />
              )}
              <circle
                cx={dotCx}
                cy={dotCy}
                r={r}
                fill={color}
                stroke="rgba(0,0,0,0.5)"
                strokeWidth={1.5}
                className="transition-all duration-150"
              />
            </g>
          );
        })}

        {/* ── Invisible hit targets for hover + click ─── */}
        {sorted.map((d, i) => (
          <rect
            key={`hit-${i}`}
            x={x(d._fx) - (plotW / sorted.length) / 2}
            y={MARGIN.top}
            width={plotW / sorted.length}
            height={plotH}
            fill="transparent"
            style={{ cursor: 'pointer' }}
            onMouseEnter={() => { if (pinnedIdx == null) setHoverIdx(i); }}
            onClick={() => setPinnedIdx((prev) => prev === i ? null : i)}
            onTouchStart={(e) => { e.preventDefault(); setPinnedIdx((prev) => prev === i ? null : i); }}
          />
        ))}

        {/* ── Transaction annotation diamonds ───── */}
        {chartTxns.map((txn, i) => {
          const cx = x(txn._fx);
          const cy = y(interpolateValue(txn._fx));
          const s = isMobile ? 4 : 5;
          return (
            <g key={`txn-${i}`}>
              <line
                x1={cx} x2={cx} y1={cy + s + 2} y2={MARGIN.top + plotH}
                stroke="rgba(30,64,175,0.15)" strokeWidth={1} strokeDasharray="3,3"
              />
              <polygon
                points={`${cx},${cy - s} ${cx + s},${cy} ${cx},${cy + s} ${cx - s},${cy}`}
                fill="white" stroke="#1d4ed8" strokeWidth={1.5}
              />
            </g>
          );
        })}

        {/* ── X-axis labels ───────────────────── */}
        {uniqueYears
          .filter((_, i) => !isMobile || i % 2 === 0 || i === uniqueYears.length - 1)
          .map((yr) => (
          <text
            key={yr}
            x={x(yr)}
            y={dims.height - 8}
            textAnchor="middle"
            className="fill-gray-500"
            fontSize={isMobile ? 9 : 11}
            fontFamily="inherit"
          >
            {isMobile ? String(yr).slice(-2) : yr}
          </text>
        ))}

        {/* ── Hover crosshair ────────────────────── */}
        {hovered && activeIdx != null && (
          <line
            x1={x(hovered._fx)}
            x2={x(hovered._fx)}
            y1={MARGIN.top}
            y2={MARGIN.top + plotH}
            stroke="rgba(0,0,0,0.10)"
            strokeWidth={1}
            strokeDasharray="4,3"
          />
        )}
      </svg>

      {/* ── Hover tooltip card ──────────────────────── */}
      {hovered && activeIdx != null && (
        <div
          className={`absolute z-20 ${pinnedIdx != null ? '' : 'pointer-events-none'}`}
          style={{
            left: isMobile ? '50%' : `${x(hovered._fx)}px`,
            top: isMobile ? '8px' : `${y(hovered.value) - 8}px`,
            transform: isMobile
              ? 'translateX(-50%)'
              : `translate(${
                  activeIdx > sorted.length * 0.7 ? 'calc(-100% - 12px)' : '12px'
                }, -100%)`,
          }}
        >
          <div className="bg-white border border-gray-200 rounded-lg shadow-xl px-4 py-3 text-xs space-y-1" style={{ minWidth: isMobile ? '180px' : '220px', maxWidth: isMobile ? '260px' : '340px' }}>
            {/* Navigation header */}
            <div className="flex items-center justify-between gap-2 -mt-1 -mx-1 mb-1">
              <button
                onClick={(e) => { e.stopPropagation(); goPrev(); }}
                disabled={!canPrev}
                className={`p-1 rounded transition-colors ${
                  canPrev ? 'text-gray-500 hover:text-gray-800 hover:bg-gray-100' : 'text-gray-200 cursor-default'
                }`}
                aria-label="Previous"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6" /></svg>
              </button>
              <span className="text-[10px] text-gray-400 tabular-nums">{activeIdx + 1} / {sorted.length}</span>
              <button
                onClick={(e) => { e.stopPropagation(); goNext(); }}
                disabled={!canNext}
                className={`p-1 rounded transition-colors ${
                  canNext ? 'text-gray-500 hover:text-gray-800 hover:bg-gray-100' : 'text-gray-200 cursor-default'
                }`}
                aria-label="Next"
              >
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6" /></svg>
              </button>
            </div>

            {/* Trade value */}
            <div className="font-semibold text-gray-900 text-sm">{fmtDollarFull(hovered.value)}</div>

            {/* Transaction type (only if present) */}
            {hovered.transactionType && (
              <div className="text-amber-600 font-medium text-[11px]">{txnLabel(hovered.transactionType)}</div>
            )}

            {/* Date */}
            <div className="text-gray-400 text-[11px]">{hovered.date ?? hovered.year}</div>

            {/* Label — full text, word-wrapped */}
            {hovered.label && (
              <div className="text-gray-600 text-[11px] leading-snug" style={{ wordBreak: 'break-word' }}>
                {hovered.label}
              </div>
            )}

            {/* ── Visual years of control bar ── */}
            {hovered.yearsControl != null && hovered.yearsControl > 0 && (() => {
              const yrs = Math.round(hovered.yearsControl!);
              const baseYear = hovered.year;
              const cells = Array.from({ length: yrs }, (_, i) => baseYear + i);
              return (
                <div className="pt-1 mt-1 border-t border-gray-100">
                  <div className="flex justify-between items-baseline mb-1">
                    <span className="text-[10px] text-gray-400 font-medium">Contract Control</span>
                    <span className="text-[10px] text-gray-400">{yrs} yr{yrs > 1 ? 's' : ''}</span>
                  </div>
                  <div className="flex gap-0.5">
                    {cells.map((yr, i) => (
                      <div key={yr} className="flex-1 flex flex-col items-center gap-0.5">
                        <div
                          className="w-full h-4 rounded-sm"
                          style={{
                            backgroundColor: i === 0
                              ? (teamColor + 'FF')
                              : (teamColor + '70'),
                          }}
                        />
                        <span className="text-[8px] text-gray-400 tabular-nums">
                          {String(yr).slice(-2)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              );
            })()}

            {/* ── Metadata from projections ── */}
            {(hovered.projectedWar != null || hovered.projectedSalary != null || hovered.warPerYear != null) && (
              <div className="border-t border-gray-100 pt-1 mt-1 space-y-0.5 text-[11px]">
                {hovered.projectedWar != null && (
                  <div className="flex justify-between gap-4">
                    <span className="text-gray-400">Projected WAR</span>
                    <span className="text-gray-600 font-medium">{fmtWar(hovered.projectedWar)}</span>
                  </div>
                )}
                {hovered.warPerYear != null && (
                  <div className="flex justify-between gap-4">
                    <span className="text-gray-400">WAR / Year</span>
                    <span className="text-gray-600 font-medium">{fmtWar(hovered.warPerYear)}</span>
                  </div>
                )}
                {hovered.projectedSalary != null && (
                  <div className="flex justify-between gap-4">
                    <span className="text-gray-400">Projected Salary</span>
                    <span className="text-gray-600 font-medium">{fmtDollarFull(hovered.projectedSalary)}</span>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      )}

      {/* ── Legend ─── */}
      <div className="flex flex-wrap gap-4 mt-3 px-1">
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: DOT_COLOR.prospect }} />
          <span className="text-[10px] text-gray-500 font-medium">Prospect</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: DOT_COLOR.mlb_surplus }} />
          <span className="text-[10px] text-gray-500 font-medium">Trade Value</span>
        </div>
        <div className="flex items-center gap-1.5">
          <span className="w-2.5 h-2.5 rounded-full inline-block" style={{ backgroundColor: DOT_COLOR.free_agent }} />
          <span className="text-[10px] text-gray-500 font-medium">Free Agent</span>
        </div>
        {chartTxns.length > 0 && (
          <div className="flex items-center gap-1.5">
            <svg width="10" height="10" viewBox="0 0 10 10">
              <polygon points="5,0 10,5 5,10 0,5" fill="white" stroke="#1d4ed8" strokeWidth="1.5" />
            </svg>
            <span className="text-[10px] text-gray-500 font-medium">Transaction</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default TradeValueChart;

