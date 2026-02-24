import React, { useState, useRef, useEffect, useMemo } from 'react';
import type { TradeValuePoint } from '../services/api';

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
  prospect: { label: 'Prospect', color: '#a78bfa' },    // violet-400
  mlb_surplus: { label: 'MLB Surplus', color: '#34d399' }, // emerald-400
  projected: { label: 'Projected', color: '#60a5fa' },  // blue-400
};

// ─── Chart Dimensions ───────────────────────────────────────
const MARGIN = { top: 20, right: 20, bottom: 36, left: 60 };
const DOT_RADIUS = 5;
const HOVER_RADIUS = 7;

// ─── Component ──────────────────────────────────────────────
interface Props {
  data: TradeValuePoint[];
  teamColor: string;
  teamAccent: string;
}

const TradeValueChart: React.FC<Props> = ({ data, teamColor, teamAccent }) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const [hoverIdx, setHoverIdx] = useState<number | null>(null);
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

  // ── Projected segment dashed ────────────────────────
  let lastNonProjectedIdx = -1;
  for (let i = sorted.length - 1; i >= 0; i--) {
    if (sorted[i].valueType !== 'projected') { lastNonProjectedIdx = i; break; }
  }
  const projectedStart = lastNonProjectedIdx >= 0 ? lastNonProjectedIdx : 0;

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
          {/* Projected area gradient (lighter) */}
          <linearGradient id="tvProjGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={teamColor} stopOpacity={0.12} />
            <stop offset="100%" stopColor={teamColor} stopOpacity={0.01} />
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

        {/* ── Solid line (non-projected) ──────── */}
        {projectedStart > 0 && (
          <path
            d={`M${sorted.slice(0, projectedStart + 1).map((d) => `${x(d.year)},${y(d.value)}`).join('L')}`}
            fill="none"
            stroke={teamAccent}
            strokeWidth={2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}

        {/* ── Dashed line for projected segment ─ */}
        {projectedStart < sorted.length - 1 && (
          <path
            d={`M${sorted.slice(projectedStart).map((d) => `${x(d.year)},${y(d.value)}`).join('L')}`}
            fill="none"
            stroke={teamAccent}
            strokeWidth={2}
            strokeLinejoin="round"
            strokeLinecap="round"
            strokeDasharray="6,4"
            opacity={0.6}
          />
        )}

        {/* If all points are non-projected, draw full solid line */}
        {projectedStart === sorted.length - 1 && sorted.every(d => d.valueType !== 'projected') && (
          <path
            d={linePath}
            fill="none"
            stroke={teamAccent}
            strokeWidth={2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        )}

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
        {presentTypes.includes('projected') && (
          <div className="flex items-center gap-1.5">
            <span className="inline-block w-4 border-t-2 border-dashed" style={{ borderColor: teamAccent, opacity: 0.6 }} />
            <span className="text-[10px] text-surface-500 font-medium">Projected</span>
          </div>
        )}
      </div>
    </div>
  );
};

export default TradeValueChart;
