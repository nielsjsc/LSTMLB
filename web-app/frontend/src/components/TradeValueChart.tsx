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

const TYPE_META: Record<string, { label: string; color: string }> = {
  prospect: { label: 'Prospect', color: '#a78bfa' },       // violet-400
  mlb_surplus: { label: 'Trade Value', color: '#34d399' },  // emerald-400
};

// Transaction type codes that affect trade value (contract events)
const _TXN_CHART_CODES = new Set(['SGN', 'SFA', 'FA', 'TR', 'EXT']);

const _TXN_LABELS: Record<string, string> = {
  SGN: 'Signed',
  SFA: 'Signed (FA)',
  FA: 'Free Agency',
  TR: 'Traded',
  EXT: 'Extension',
};

const _TXN_COLOR = '#f59e0b'; // amber-500
const _EXT_COLOR = '#38bdf8'; // sky-400  — contract extensions

// ─── Chart Dimensions ───────────────────────────────────────
const MARGIN = { top: 20, right: 20, bottom: 36, left: 60 };
const DOT_RADIUS = 5;
const HOVER_RADIUS = 7;

// ─── Transaction event processed for chart ──────────────────
interface ChartTxn {
  year: number;       // fractional year (e.g. 2021.5 for July)
  label: string;
  shortDesc: string;
  typeCode: string;   // original type code (EXT, SGN, SFA, etc.)
}

// ─── Component ──────────────────────────────────────────────
interface Props {
  data: TradeValuePoint[];
  teamColor: string;
  teamAccent: string;
  transactions?: Transaction[];
}

const TradeValueChart: React.FC<Props> = ({ data, teamColor, teamAccent, transactions }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
  const [hoverTxn, setHoverTxn] = useState<number | null>(null);
  const [dims, setDims] = useState({ width: 700, height: 300 });

  // Responsive width via ResizeObserver
  useEffect(() => {
    const node = containerRef.current;
    if (!node) return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        if (w > 0) setDims({ width: w, height: Math.min(320, Math.max(220, w * 0.4)) });
      }
    });
    ro.observe(node);
    return () => ro.disconnect();
  }, []);

  // Sort by year
  const sorted = useMemo(() => [...data].sort((a, b) => a.year - b.year), [data]);

  // Process transactions for chart markers
  const chartTxns = useMemo<ChartTxn[]>(() => {
    if (!transactions || transactions.length === 0) return [];
    const seen = new Set<string>();
    return transactions
      .filter((t) => _TXN_CHART_CODES.has(t.typeCode))
      .map((t) => {
        const d = new Date(t.date + 'T12:00:00');
        const yr = d.getFullYear() + (d.getMonth() / 12);
        const code = t.typeCode;
        // For signings, extract short description from description text
        let shortDesc = _TXN_LABELS[code] || t.typeDesc;
        // Try to extract dollar amount for signings
        const moneyMatch = t.description.match(/\$[\d,.]+\s*(million|billion)?/i);
        if (moneyMatch) shortDesc += ` (${moneyMatch[0]})`;
        // Add team info
        if (t.toTeam) shortDesc += ` → ${t.toTeam}`;
        return { year: yr, label: _TXN_LABELS[code] || code, shortDesc, typeCode: code };
      })
      .filter((t) => {
        // Deduplicate very close events
        const key = `${Math.round(t.year)}-${t.label}`;
        if (seen.has(key)) return false;
        seen.add(key);
        return true;
      })
      .sort((a, b) => a.year - b.year);
  }, [transactions]);

  if (sorted.length === 0) return null;

  const plotW = dims.width - MARGIN.left - MARGIN.right;
  const plotH = dims.height - MARGIN.top - MARGIN.bottom;

  // ── Scales ──────────────────────────────────────────
  const years = sorted.map((d) => d.year);
  const minYear = years[0];
  const maxYear = years[years.length - 1];
  const yearSpan = Math.max(maxYear - minYear, 1);

  const values = sorted.map((d) => d.value);
  const rawMax = Math.max(...values);
  const rawMin = Math.min(0, ...values); // always include 0
  const valRange = rawMax - rawMin || 1;
  const yPad = valRange * 0.12;
  const yMax = rawMax + yPad;
  const yMin = rawMin - (rawMin < 0 ? yPad : 0);
  const yRange = yMax - yMin || 1;

  const x = (year: number) => MARGIN.left + ((year - minYear) / yearSpan) * plotW;
  const y = (val: number) => MARGIN.top + (1 - (val - yMin) / yRange) * plotH;

  // ── Path (line + area) ──────────────────────────────
  const linePoints = sorted.map((d) => `${x(d.year)},${y(d.value)}`);
  const linePath = `M${linePoints.join('L')}`;
  const areaPath = `${linePath}L${x(maxYear)},${y(0)}L${x(minYear)},${y(0)}Z`;

  // ── Gridlines ───────────────────────────────────────
  const yTicks = useMemo(() => {
    const count = 5;
    const step = yRange / count;
    const niceStep = Math.pow(10, Math.floor(Math.log10(step))) * Math.round(step / Math.pow(10, Math.floor(Math.log10(step))));
    const ticks: number[] = [];
    let t = Math.ceil(yMin / niceStep) * niceStep;
    while (t <= yMax) {
      ticks.push(t);
      t += niceStep;
    }
    return ticks.length > 0 ? ticks : [0];
  }, [yMin, yMax, yRange]);

  // Build tooltip
  const hovered = hoverIdx != null ? sorted[hoverIdx] : null;

  // Find unique value types present
  const presentTypes = useMemo(
    () => [...new Set(sorted.map((d) => d.valueType))],
    [sorted],
  );

  return (
    <div ref={containerRef} className="w-full relative">
      <svg
        ref={svgRef}
        width={dims.width}
        height={dims.height}
        className="select-none"
        onMouseLeave={() => setHoverIdx(null)}
      >
        <defs>
          {/* Area gradient */}
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
              stroke="rgba(255,255,255,0.06)"
              strokeWidth={1}
            />
            <text
              x={MARGIN.left - 8}
              y={y(t)}
              textAnchor="end"
              dominantBaseline="middle"
              className="fill-surface-500"
              fontSize={11}
              fontFamily="inherit"
            >
              {fmtDollar(t)}
            </text>
          </g>
        ))}

        {/* ── Zero line (if visible) ──────────── */}
        {yMin < 0 && (
          <line
            x1={MARGIN.left}
            x2={dims.width - MARGIN.right}
            y1={y(0)}
            y2={y(0)}
            stroke="rgba(255,255,255,0.12)"
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

        {/* ── Data dots ───────────────────────── */}
        {sorted.map((d, i) => {
          const meta = TYPE_META[d.valueType] ?? TYPE_META.mlb_surplus;
          const isHovered = hoverIdx === i;
          return (
            <g key={`${d.year}-${d.valueType}`}>
              {/* Outer glow on hover */}
              {isHovered && (
                <circle
                  cx={x(d.year)}
                  cy={y(d.value)}
                  r={HOVER_RADIUS + 4}
                  fill={meta.color}
                  opacity={0.18}
                />
              )}
              <circle
                cx={x(d.year)}
                cy={y(d.value)}
                r={isHovered ? HOVER_RADIUS : DOT_RADIUS}
                fill={meta.color}
                stroke="rgba(0,0,0,0.5)"
                strokeWidth={1.5}
                className="transition-all duration-150"
              />
            </g>
          );
        })}

        {/* ── Transaction markers (vertical lines + diamond) ─── */}
        {chartTxns
          .filter((t) => t.year >= minYear && t.year <= maxYear + 0.5)
          .map((t, i) => {
            const tx = x(t.year);
            const isHov = hoverTxn === i;
            const isContract = t.typeCode === 'EXT' || t.typeCode === 'SFA';
            const markerColor = isContract ? _EXT_COLOR : _TXN_COLOR;
            return (
              <g key={`txn-${i}`}>
                {/* Dashed vertical line */}
                <line
                  x1={tx}
                  x2={tx}
                  y1={MARGIN.top}
                  y2={MARGIN.top + plotH}
                  stroke={markerColor}
                  strokeWidth={isHov ? 1.5 : 1}
                  strokeDasharray="3,4"
                  opacity={isHov ? 0.7 : 0.35}
                />
                {/* Diamond marker at bottom */}
                <polygon
                  points={`${tx},${MARGIN.top + plotH - 6} ${tx + 4},${MARGIN.top + plotH} ${tx},${MARGIN.top + plotH + 6} ${tx - 4},${MARGIN.top + plotH}`}
                  fill={isHov ? markerColor : markerColor + '80'}
                  stroke="rgba(0,0,0,0.4)"
                  strokeWidth={1}
                />
                {/* Hit area */}
                <rect
                  x={tx - 8}
                  y={MARGIN.top}
                  width={16}
                  height={plotH}
                  fill="transparent"
                  onMouseEnter={() => { setHoverTxn(i); setHoverIdx(null); }}
                  onMouseLeave={() => setHoverTxn(null)}
                />
              </g>
            );
          })}

        {/* ── Invisible hit targets for hover ─── */}
        {sorted.map((d, i) => (
          <rect
            key={`hit-${d.year}-${i}`}
            x={x(d.year) - (plotW / sorted.length) / 2}
            y={MARGIN.top}
            width={plotW / sorted.length}
            height={plotH}
            fill="transparent"
            onMouseEnter={() => setHoverIdx(i)}
            onTouchStart={() => setHoverIdx(i)}
          />
        ))}

        {/* ── X-axis labels ───────────────────── */}
        {years.map((yr) => (
          <text
            key={yr}
            x={x(yr)}
            y={dims.height - 8}
            textAnchor="middle"
            className="fill-surface-500"
            fontSize={11}
            fontFamily="inherit"
          >
            {yr}
          </text>
        ))}

        {/* ── Hover crosshair + tooltip ────────── */}
        {hovered && hoverIdx != null && (
          <>
            <line
              x1={x(hovered.year)}
              x2={x(hovered.year)}
              y1={MARGIN.top}
              y2={MARGIN.top + plotH}
              stroke="rgba(255,255,255,0.15)"
              strokeWidth={1}
              strokeDasharray="4,3"
            />
          </>
        )}
      </svg>

      {/* ── Floating tooltip (HTML for better styling) ──────── */}
      {hovered && hoverIdx != null && (
        <div
          className="pointer-events-none absolute z-20"
          style={{
            left: `${x(hovered.year)}px`,
            top: `${y(hovered.value) - 8}px`,
            transform: `translate(${
              hoverIdx > sorted.length * 0.7 ? 'calc(-100% - 12px)' : '12px'
            }, -100%)`,
          }}
        >
          <div className="bg-surface-800 border border-white/10 rounded-lg shadow-xl px-3 py-2 text-xs whitespace-nowrap">
            <div className="font-semibold text-white mb-0.5">{fmtDollarFull(hovered.value)}</div>
            <div className="text-surface-400">{hovered.year} · {hovered.label}</div>
          </div>
        </div>
      )}

      {/* ── Transaction hover tooltip ──────────── */}
      {hoverTxn != null && chartTxns[hoverTxn] && (() => {
        const t = chartTxns[hoverTxn];
        const tx = x(t.year);
        const isContract = t.typeCode === 'EXT' || t.typeCode === 'SFA';
        const tooltipBorder = isContract ? 'border-sky-400/20' : 'border-amber-500/20';
        const tooltipText = isContract ? 'text-sky-400' : 'text-amber-400';
        return (
          <div
            className="pointer-events-none absolute z-20"
            style={{
              left: `${tx}px`,
              top: `${MARGIN.top + 12}px`,
              transform: `translate(${tx > dims.width * 0.65 ? 'calc(-100% - 12px)' : '12px'}, 0)`,
            }}
          >
            <div className={`bg-surface-800 border ${tooltipBorder} rounded-lg shadow-xl px-3 py-2 text-xs whitespace-nowrap`}>
              <div className={`font-semibold ${tooltipText} mb-0.5`}>{t.label}</div>
              <div className="text-surface-400 max-w-[220px] truncate">{t.shortDesc}</div>
            </div>
          </div>
        );
      })()}

      {/* ── Legend ─────────────────────────────── */}
      <div className="flex flex-wrap gap-4 mt-3 px-1">
        {presentTypes.map((type) => {
          const meta = TYPE_META[type];
          if (!meta) return null;
          return (
            <div key={type} className="flex items-center gap-1.5">
              <span
                className="w-2.5 h-2.5 rounded-full inline-block"
                style={{ backgroundColor: meta.color }}
              />
              <span className="text-[10px] text-surface-500 font-medium">{meta.label}</span>
            </div>
          );
        })}
        {chartTxns.length > 0 && (
          <>
            {chartTxns.some((t) => t.typeCode !== 'EXT' && t.typeCode !== 'SFA') && (
              <div className="flex items-center gap-1.5">
                <span
                  className="inline-block"
                  style={{ width: 10, height: 10, backgroundColor: _TXN_COLOR, clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)' }}
                />
                <span className="text-[10px] text-surface-500 font-medium">Transaction</span>
              </div>
            )}
            {chartTxns.some((t) => t.typeCode === 'EXT' || t.typeCode === 'SFA') && (
              <div className="flex items-center gap-1.5">
                <span
                  className="inline-block"
                  style={{ width: 10, height: 10, backgroundColor: _EXT_COLOR, clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)' }}
                />
                <span className="text-[10px] text-surface-500 font-medium">Contract</span>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default TradeValueChart;
