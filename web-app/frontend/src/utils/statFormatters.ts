export type StatFormatterType = 
  | 'percent' 
  | 'decimal' 
  | 'war' 
  | 'money' 
  | 'integer'
  | 'year'
  | 'string'
  | 'number'
  | 'average';

// Basic Formatters
export const formatString = (value: any): string => {
  if (value === null || value === undefined) return '-';
  return String(value);
};

export const formatNumber = (value: number | null): string => {
  if (value === null || isNaN(value)) return '-';
  return value.toLocaleString();
};

export const formatInteger = (value: number | null): string => {
  if (value === null || isNaN(value)) return '-';
  return Math.round(value).toString();
};

export const formatYear = (value: number | null): string => {
  if (value === null || isNaN(value)) return '-';
  return value.toString();
};

// Baseball Specific Formatters
export const formatPercent = (value: number | null): string => {
  if (value === null || isNaN(value)) return '-';
  return `${(value * 100).toFixed(1)}%`;
};

export const formatDecimal = (value: number | null): string => {
  if (value === null || isNaN(value)) return '-';
  return value.toFixed(3);
};

export const formatAverage = (value: number | null): string => {
  if (value === null || isNaN(value)) return '-';
  return value.toFixed(3).substring(1); // Removes leading 0 for batting averages
};

export const formatWAR = (value: number | null): string => {
  if (value === null || isNaN(value)) return '-';
  return value.toFixed(1);
};

// Value Formatters
export const formatMoney = (value: number | null): string => {
  if (value === null || isNaN(value)) return '-';
  const millions = value / 1000000;
  return millions >= 0 
    ? `$${millions.toFixed(1)}M`
    : `-$${Math.abs(millions).toFixed(1)}M`;
};

export const formatters: Record<StatFormatterType, (value: any) => string> = {
  string: formatString,
  number: formatNumber,
  integer: formatInteger,
  year: formatYear,
  percent: formatPercent,
  decimal: formatDecimal,
  average: formatAverage,
  war: formatWAR,
  money: formatMoney
};