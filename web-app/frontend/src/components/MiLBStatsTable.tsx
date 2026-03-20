import React, { useState, useMemo } from 'react';
import type { MiLBHittingSeason, MiLBPitchingSeason, MiLBStatsResponse } from '../services/api';

// ── Level ordering & badge colours ──────────────────────────────────────────

const LEVEL_ORDER: Record<string, number> = {
  DSL: 0, FCL: 1, CPX: 2, R: 3, Rk: 3, 'A-': 4,
  'A (Short)': 5, A: 6, 'A+': 7, AA: 8, AAA: 9, MLB: 10,
};

const levelOrder = (lvl: string) => LEVEL_ORDER[lvl] ?? 20;

const LEVEL_COLORS: Record<string, string> = {
  MLB: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/25',
  AAA: 'bg-blue-500/15 text-blue-400 border-blue-500/25',
  AA: 'bg-sky-500/15 text-sky-400 border-sky-500/25',
  'A+': 'bg-violet-500/15 text-violet-400 border-violet-500/25',
  A: 'bg-amber-500/15 text-amber-400 border-amber-500/25',
  'A-': 'bg-orange-500/15 text-orange-400 border-orange-500/25',
};

const levelBadge = (lvl: string) => LEVEL_COLORS[lvl] ?? 'bg-gray-100/60 text-gray-500 border-gray-200';

// ── Stat formatters ─────────────────────────────────────────────────────────

const pct = (v: number | null) => (v == null ? '-' : `${v.toFixed(1)}%`);
const dec3 = (v: number | null) => (v == null ? '-' : v.toFixed(3).replace(/^0/, ''));
const dec2 = (v: number | null) => (v == null ? '-' : v.toFixed(2));
const intStat = (v: number | null) => (v == null ? '-' : Math.round(v).toString());

const wrcColor = (v: number | null) => {
  if (v == null) return 'text-gray-500';
  if (v >= 140) return 'text-emerald-400';
  if (v >= 115) return 'text-green-400';
  if (v >= 85) return 'text-gray-800';
  if (v >= 70) return 'text-amber-400';
  return 'text-red-400';
};

const eraColor = (v: number | null) => {
  if (v == null) return 'text-gray-500';
  if (v <= 2.5) return 'text-emerald-400';
  if (v <= 3.5) return 'text-green-400';
  if (v <= 4.5) return 'text-gray-800';
  if (v <= 5.5) return 'text-amber-400';
  return 'text-red-400';
};

// ── Year-group types ────────────────────────────────────────────────────────

interface YearGroup<T> {
  season: number;
  rows: T[];
}

function groupBySeason<T extends { season: number; level: string }>(data: T[]): YearGroup<T>[] {
  const sorted = [...data].sort(
    (a, b) => b.season - a.season || levelOrder(a.level) - levelOrder(b.level),
  );
  const groups: YearGroup<T>[] = [];
  let cur: YearGroup<T> | null = null;
  for (const row of sorted) {
    if (!cur || cur.season !== row.season) {
      cur = { season: row.season, rows: [] };
      groups.push(cur);
    }
    cur.rows.push(row);
  }
  return groups;
}

// ── Chevron icon ────────────────────────────────────────────────────────────

const Chevron: React.FC<{ open: boolean }> = ({ open }) => (
  <svg
    className={`w-3.5 h-3.5 text-gray-400 transition-transform duration-200 ${open ? 'rotate-180' : ''}`}
    fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}
  >
    <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
  </svg>
);

// ── Hitting year group ──────────────────────────────────────────────────────

const HITTING_COLS = ['Year', 'Tm', 'Lv', 'Age', 'PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BB%', 'K%', 'wOBA', 'wRC+', 'Spd', 'BABIP'] as const;
const HITTING_RIGHT = new Set(['PA', 'AVG', 'OBP', 'SLG', 'OPS', 'ISO', 'BB%', 'K%', 'wOBA', 'wRC+', 'Spd', 'BABIP']);

const HittingYearGroup: React.FC<{ group: YearGroup<MiLBHittingSeason>; defaultOpen: boolean }> = ({
  group,
  defaultOpen,
}) => {
  const [open, setOpen] = useState(defaultOpen);
  const multiLevel = group.rows.length > 1;

  return (
    <>
      {/* Year header row */}
      <tr
        className="cursor-pointer hover:bg-white/[0.03] transition-colors select-none"
        onClick={() => setOpen(!open)}
      >
        <td className="px-2 py-2 text-gray-900 font-semibold text-[12px]">
          <span className="inline-flex items-center gap-1.5">
            <Chevron open={open} />
            {group.season}
          </span>
        </td>
        {/* Summary cells: show dash for multi-level, or inline values for single */}
        {!multiLevel && group.rows[0] ? (
          <>
            <td className="px-2 py-2 text-gray-500 text-[11px] whitespace-nowrap">{group.rows[0].team}</td>
            <td className="px-2 py-2 text-[11px]">
              <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-semibold border ${levelBadge(group.rows[0].level)}`}>
                {group.rows[0].level}
              </span>
            </td>
            <td className="px-2 py-2 text-gray-500 text-[11px]">{group.rows[0].age}</td>
            <td className="px-2 py-2 text-gray-600 text-right font-mono text-[11px]">{group.rows[0].pa}</td>
            <td className="px-2 py-2 text-gray-800 text-right font-mono text-[11px]">{dec3(group.rows[0].avg)}</td>
            <td className="px-2 py-2 text-gray-800 text-right font-mono text-[11px]">{dec3(group.rows[0].obp)}</td>
            <td className="px-2 py-2 text-gray-800 text-right font-mono text-[11px]">{dec3(group.rows[0].slg)}</td>
            <td className="px-2 py-2 text-gray-900 text-right font-mono font-semibold text-[11px]">{dec3(group.rows[0].ops)}</td>
            <td className="px-2 py-2 text-gray-600 text-right font-mono text-[11px]">{dec3(group.rows[0].iso)}</td>
            <td className="px-2 py-2 text-gray-600 text-right font-mono text-[11px]">{pct(group.rows[0].bb_pct)}</td>
            <td className="px-2 py-2 text-gray-600 text-right font-mono text-[11px]">{pct(group.rows[0].k_pct)}</td>
            <td className="px-2 py-2 text-gray-800 text-right font-mono text-[11px]">{dec3(group.rows[0].woba)}</td>
            <td className={`px-2 py-2 text-right font-mono font-semibold text-[11px] ${wrcColor(group.rows[0].wrc_plus)}`}>{intStat(group.rows[0].wrc_plus)}</td>
            <td className="px-2 py-2 text-gray-500 text-right font-mono text-[11px]">{group.rows[0].spd != null ? dec2(group.rows[0].spd) : '-'}</td>
            <td className="px-2 py-2 text-gray-500 text-right font-mono text-[11px]">{dec3(group.rows[0].babip)}</td>
          </>
        ) : (
          <>
            <td className="px-2 py-2 text-gray-400 text-[11px]" colSpan={3}>
              {group.rows.length} levels
            </td>
            {/* Empty cells to fill the row */}
            <td colSpan={11} />
          </>
        )}
      </tr>

      {/* Expanded level rows */}
      {open && multiLevel && group.rows.map((h, j) => (
        <tr
          key={`${h.season}-${h.team}-${h.level}`}
          className={`text-[11px] border-b border-white/[0.02] ${j % 2 === 0 ? 'bg-gray-50/50' : 'bg-white/[0.03]'}`}
        >
          <td className="px-2 py-1.5 pl-8 text-gray-400 font-mono text-[10px]" />
          <td className="px-2 py-1.5 text-gray-500 whitespace-nowrap">{h.team}</td>
          <td className="px-2 py-1.5">
            <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-semibold border ${levelBadge(h.level)}`}>
              {h.level}
            </span>
          </td>
          <td className="px-2 py-1.5 text-gray-500">{h.age}</td>
          <td className="px-2 py-1.5 text-gray-600 text-right font-mono">{h.pa}</td>
          <td className="px-2 py-1.5 text-gray-800 text-right font-mono">{dec3(h.avg)}</td>
          <td className="px-2 py-1.5 text-gray-800 text-right font-mono">{dec3(h.obp)}</td>
          <td className="px-2 py-1.5 text-gray-800 text-right font-mono">{dec3(h.slg)}</td>
          <td className="px-2 py-1.5 text-gray-900 text-right font-mono font-semibold">{dec3(h.ops)}</td>
          <td className="px-2 py-1.5 text-gray-600 text-right font-mono">{dec3(h.iso)}</td>
          <td className="px-2 py-1.5 text-gray-600 text-right font-mono">{pct(h.bb_pct)}</td>
          <td className="px-2 py-1.5 text-gray-600 text-right font-mono">{pct(h.k_pct)}</td>
          <td className="px-2 py-1.5 text-gray-800 text-right font-mono">{dec3(h.woba)}</td>
          <td className={`px-2 py-1.5 text-right font-mono font-semibold ${wrcColor(h.wrc_plus)}`}>{intStat(h.wrc_plus)}</td>
          <td className="px-2 py-1.5 text-gray-500 text-right font-mono">{h.spd != null ? dec2(h.spd) : '-'}</td>
          <td className="px-2 py-1.5 text-gray-500 text-right font-mono">{dec3(h.babip)}</td>
        </tr>
      ))}
    </>
  );
};

// ── Pitching year group ─────────────────────────────────────────────────────

const PITCHING_COLS = ['Year', 'Tm', 'Lv', 'Age', 'IP', 'ERA', 'FIP', 'xFIP', 'K/9', 'BB/9', 'K/BB', 'K%', 'BB%', 'WHIP', 'AVG', 'HR/9', 'BABIP'] as const;

const PitchingYearGroup: React.FC<{ group: YearGroup<MiLBPitchingSeason>; defaultOpen: boolean }> = ({
  group,
  defaultOpen,
}) => {
  const [open, setOpen] = useState(defaultOpen);
  const multiLevel = group.rows.length > 1;

  return (
    <>
      {/* Year header row */}
      <tr
        className="cursor-pointer hover:bg-white/[0.03] transition-colors select-none"
        onClick={() => setOpen(!open)}
      >
        <td className="px-2 py-2 text-gray-900 font-semibold text-[12px]">
          <span className="inline-flex items-center gap-1.5">
            <Chevron open={open} />
            {group.season}
          </span>
        </td>
        {!multiLevel && group.rows[0] ? (
          <>
            <td className="px-2 py-2 text-gray-500 text-[11px] whitespace-nowrap">{group.rows[0].team}</td>
            <td className="px-2 py-2 text-[11px]">
              <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-semibold border ${levelBadge(group.rows[0].level)}`}>
                {group.rows[0].level}
              </span>
            </td>
            <td className="px-2 py-2 text-gray-500 text-[11px] text-right">{group.rows[0].age}</td>
            <td className="px-2 py-2 text-gray-600 text-right font-mono text-[11px]">{dec2(group.rows[0].ip)}</td>
            <td className={`px-2 py-2 text-right font-mono font-semibold text-[11px] ${eraColor(group.rows[0].era)}`}>{dec2(group.rows[0].era)}</td>
            <td className={`px-2 py-2 text-right font-mono font-semibold text-[11px] ${eraColor(group.rows[0].fip)}`}>{dec2(group.rows[0].fip)}</td>
            <td className="px-2 py-2 text-gray-600 text-right font-mono text-[11px]">{dec2(group.rows[0].xfip)}</td>
            <td className="px-2 py-2 text-gray-800 text-right font-mono text-[11px]">{dec2(group.rows[0].k_9)}</td>
            <td className="px-2 py-2 text-gray-600 text-right font-mono text-[11px]">{dec2(group.rows[0].bb_9)}</td>
            <td className="px-2 py-2 text-gray-600 text-right font-mono text-[11px]">{dec2(group.rows[0].k_bb)}</td>
            <td className="px-2 py-2 text-gray-800 text-right font-mono text-[11px]">{pct(group.rows[0].k_pct)}</td>
            <td className="px-2 py-2 text-gray-600 text-right font-mono text-[11px]">{pct(group.rows[0].bb_pct)}</td>
            <td className="px-2 py-2 text-gray-800 text-right font-mono text-[11px]">{dec2(group.rows[0].whip)}</td>
            <td className="px-2 py-2 text-gray-600 text-right font-mono text-[11px]">{dec3(group.rows[0].avg)}</td>
            <td className="px-2 py-2 text-gray-500 text-right font-mono text-[11px]">{dec2(group.rows[0].hr_9)}</td>
            <td className="px-2 py-2 text-gray-500 text-right font-mono text-[11px]">{dec3(group.rows[0].babip)}</td>
          </>
        ) : (
          <>
            <td className="px-2 py-2 text-gray-400 text-[11px]" colSpan={3}>
              {group.rows.length} levels
            </td>
            <td colSpan={12} />
          </>
        )}
      </tr>

      {/* Expanded level rows */}
      {open && multiLevel && group.rows.map((p, j) => (
        <tr
          key={`${p.season}-${p.team}-${p.level}`}
          className={`text-[11px] border-b border-white/[0.02] ${j % 2 === 0 ? 'bg-gray-50/50' : 'bg-white/[0.03]'}`}
        >
          <td className="px-2 py-1.5 pl-8" />
          <td className="px-2 py-1.5 text-gray-500 whitespace-nowrap">{p.team}</td>
          <td className="px-2 py-1.5">
            <span className={`inline-block px-1.5 py-0.5 rounded text-[9px] font-semibold border ${levelBadge(p.level)}`}>
              {p.level}
            </span>
          </td>
          <td className="px-2 py-1.5 text-gray-500 text-right">{p.age}</td>
          <td className="px-2 py-1.5 text-gray-600 text-right font-mono">{dec2(p.ip)}</td>
          <td className={`px-2 py-1.5 text-right font-mono font-semibold ${eraColor(p.era)}`}>{dec2(p.era)}</td>
          <td className={`px-2 py-1.5 text-right font-mono font-semibold ${eraColor(p.fip)}`}>{dec2(p.fip)}</td>
          <td className="px-2 py-1.5 text-gray-600 text-right font-mono">{dec2(p.xfip)}</td>
          <td className="px-2 py-1.5 text-gray-800 text-right font-mono">{dec2(p.k_9)}</td>
          <td className="px-2 py-1.5 text-gray-600 text-right font-mono">{dec2(p.bb_9)}</td>
          <td className="px-2 py-1.5 text-gray-600 text-right font-mono">{dec2(p.k_bb)}</td>
          <td className="px-2 py-1.5 text-gray-800 text-right font-mono">{pct(p.k_pct)}</td>
          <td className="px-2 py-1.5 text-gray-600 text-right font-mono">{pct(p.bb_pct)}</td>
          <td className="px-2 py-1.5 text-gray-800 text-right font-mono">{dec2(p.whip)}</td>
          <td className="px-2 py-1.5 text-gray-600 text-right font-mono">{dec3(p.avg)}</td>
          <td className="px-2 py-1.5 text-gray-500 text-right font-mono">{dec2(p.hr_9)}</td>
          <td className="px-2 py-1.5 text-gray-500 text-right font-mono">{dec3(p.babip)}</td>
        </tr>
      ))}
    </>
  );
};

// ── Main export ─────────────────────────────────────────────────────────────

interface MiLBStatsTableProps {
  stats: MiLBStatsResponse;
  /** When true, most-recent year starts expanded (default: true) */
  expandLatest?: boolean;
}

const MiLBStatsTable: React.FC<MiLBStatsTableProps> = ({ stats, expandLatest = true }) => {
  const hittingGroups = useMemo(() => groupBySeason(stats.hitting), [stats.hitting]);
  const pitchingGroups = useMemo(() => groupBySeason(stats.pitching), [stats.pitching]);

  if (!hittingGroups.length && !pitchingGroups.length) return null;

  return (
    <div className="space-y-6">
      {/* ── Hitting ─────────────────────────────────────────────── */}
      {hittingGroups.length > 0 && (
        <div>
          <h3 className="text-[11px] uppercase tracking-wider text-gray-400 mb-3 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-blue-400" />
            Hitting
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  {HITTING_COLS.map((col) => (
                    <th
                      key={col}
                      className={`px-2 py-1.5 text-[10px] text-gray-400 uppercase tracking-wider whitespace-nowrap ${
                        HITTING_RIGHT.has(col) ? 'text-right' : 'text-left'
                      }`}
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {hittingGroups.map((g, idx) => (
                  <HittingYearGroup
                    key={g.season}
                    group={g}
                    defaultOpen={expandLatest && idx === 0}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* ── Pitching ───────────────────────────────────────────── */}
      {pitchingGroups.length > 0 && (
        <div>
          <h3 className="text-[11px] uppercase tracking-wider text-gray-400 mb-3 flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full bg-purple-400" />
            Pitching
          </h3>
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  {PITCHING_COLS.map((col) => (
                    <th
                      key={col}
                      className={`px-2 py-1.5 text-[10px] text-gray-400 uppercase tracking-wider whitespace-nowrap ${
                        col === 'Year' || col === 'Tm' || col === 'Lv' ? 'text-left' : 'text-right'
                      }`}
                    >
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {pitchingGroups.map((g, idx) => (
                  <PitchingYearGroup
                    key={g.season}
                    group={g}
                    defaultOpen={expandLatest && idx === 0}
                  />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default MiLBStatsTable;
