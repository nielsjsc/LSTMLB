import React from 'react';

export interface ControlYearProjection {
  year: number;
  status: string;
}

export interface ControlYearsProps {
  currentYear: number;
  /** Years of team control remaining, starting this season. */
  yearsControl?: number | null;
  /** Year the player is projected to first reach free agency. */
  probableFaYear?: number | null;
  /** Year the player is contractually guaranteed to reach free agency at the latest. */
  faYear?: number | null;
  /** Per-year status labels (e.g. "Arb 2", "Club Option"), same shape as the player details page. */
  projections?: ControlYearProjection[];
  /** Collapse into a "+N" overflow marker past this many year-cells. */
  maxCells?: number;
}

const DEFAULT_MAX_CELLS = 6;

/**
 * Compact, table-row-friendly version of the contract-control timeline used
 * on the player details page: a short strip of year cells (filled = under
 * control, dashed = option year, amber = probable free agency) plus a
 * one-line "N yrs control" caption.
 *
 * NOTE: years_control / probable_fa_year / fa_year / projections are not
 * currently part of the TradeAnalysis asset type returned by
 * services/api.ts. Add them to the analyzeTrade response and the Player
 * type (mirroring the fields already used on the player details page) to
 * populate this. Until then, assets without this data render as "—".
 */
const ControlYears: React.FC<ControlYearsProps> = ({
  currentYear,
  yearsControl,
  probableFaYear,
  faYear,
  projections = [],
  maxCells = DEFAULT_MAX_CELLS,
}) => {
  const hasData = yearsControl != null || projections.length > 0;

  if (!hasData) {
    return <span className="text-xs text-gray-300">—</span>;
  }

  const yrsCtrl = Math.max(yearsControl ?? 0, 0);
  const endYear = Math.max(
    currentYear + Math.max(yrsCtrl - 1, 0),
    probableFaYear ?? currentYear,
    currentYear
  );
  const allYears = Array.from({ length: endYear - currentYear + 1 }, (_, i) => currentYear + i);
  const overflow = allYears.length - maxCells;
  const years = overflow > 0 ? allYears.slice(0, maxCells) : allYears;

  const label =
    yrsCtrl > 0
      ? `${yrsCtrl} yr${yrsCtrl === 1 ? '' : 's'} control${faYear ? ` \u00b7 FA ${faYear}` : ''}`
      : 'Free agent';

  return (
    <div className="flex flex-col gap-1 min-w-[90px]">
      <div className="flex items-center gap-[2px]">
        {years.map((yr) => {
          const isControlled = yr < currentYear + yrsCtrl;
          const proj = projections.find((p) => p.year === yr);
          const status = (proj?.status || '').toLowerCase();
          const isOption = isControlled && status.includes('option');
          const isProbableFa = yr === probableFaYear;
          const isBeyondProbableFa = probableFaYear != null && yr > probableFaYear && !isControlled;

          let cellClass = 'bg-gray-50 border border-gray-200'; // uncontracted / unknown
          if (isOption) {
            cellClass = 'bg-transparent border border-dashed border-amber-400';
          } else if (isControlled) {
            cellClass = 'bg-gray-700 border border-gray-700';
          } else if (isProbableFa) {
            cellClass = 'bg-amber-100 border border-amber-300';
          } else if (isBeyondProbableFa) {
            cellClass = 'bg-gray-100 border border-gray-200';
          }

          return (
            <div
              key={yr}
              title={`${yr}${proj?.status ? ` \u00b7 ${proj.status}` : ''}`}
              className={`h-3.5 w-3 rounded-sm shrink-0 ${cellClass} ${
                yr === currentYear ? 'ring-1 ring-inset ring-gray-900/40' : ''
              }`}
            />
          );
        })}
        {overflow > 0 && <span className="text-[9px] text-gray-400 pl-0.5">+{overflow}</span>}
      </div>
      <span className="text-[10px] text-gray-500 tabular-nums leading-none">{label}</span>
    </div>
  );
};

export default ControlYears;