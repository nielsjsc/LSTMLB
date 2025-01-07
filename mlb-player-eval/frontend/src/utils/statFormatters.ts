export const formatPercent = (value: number | null): string => {
    if (value === null || isNaN(value)) return '-';
    return `${(value * 100).toFixed(1)}%`;
  };
  
  export const formatDecimal = (value: number | null): string => {
    if (value === null || isNaN(value)) return '-';
    return value.toFixed(3);
  };
  
  export const formatWAR = (value: number | null): string => {
    if (value === null || isNaN(value)) return '-';
    return value.toFixed(1);
  };
  
  export const formatMoney = (value: number | null): string => {
    if (value === null || isNaN(value)) return '-';
    return `$${(value / 1000000).toFixed(1)}M`;
  };
  
  export const formatInteger = (value: number | null): string => {
    if (value === null || isNaN(value)) return '-';
    return Math.round(value).toString();
  };
  
  export const formatYear = (value: number | null): string => {
    if (value === null || isNaN(value)) return '-';
    return value.toString();
  };
  
  export type StatFormatterType = 'percent' | 'decimal' | 'war' | 'money' | 'integer' | 'year';
  
  export const formatters: Record<StatFormatterType, (value: number | null) => string> = {
    percent: formatPercent,
    decimal: formatDecimal,
    war: formatWAR,
    money: formatMoney,
    integer: formatInteger,
    year: formatYear
  };