import React, { useState } from 'react';
import { formatters, StatFormatterType } from '../../../utils/statFormatters';

export interface Column {
  key: string;
  header: string;
  formatter?: StatFormatterType;
  align?: 'left' | 'center' | 'right';
  renderCell?: (value: any) => React.ReactNode;
}

interface StatTableProps {
  data: any[];
  columns: Column[];
  defaultSort?: string;
  defaultSortDirection?: 'asc' | 'desc';
  dividerYear?: number;  // Added prop
  className?: string;
}

const StatTable: React.FC<StatTableProps> = ({
  data,
  columns,
  defaultSort,
  defaultSortDirection = 'asc',
  dividerYear,  // Added prop
  className = ''
}) => {
  const [sortColumn, setSortColumn] = useState(defaultSort || columns[0].key);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>(defaultSortDirection);

  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      setSortDirection('desc');
    }
  };

  const sortedData = React.useMemo(() => {
    return [...data].sort((a, b) => {
      const aValue = a[sortColumn];
      const bValue = b[sortColumn];
      if (aValue === bValue) return 0;
      if (aValue === null || aValue === undefined) return 1;
      if (bValue === null || bValue === undefined) return -1;
      return sortDirection === 'asc' 
        ? (aValue > bValue ? 1 : -1)
        : (aValue < bValue ? 1 : -1);
    });
  }, [data, sortColumn, sortDirection]);
  const renderTableRows = () => {
    let previousYear = 0;
    return sortedData.map((row, idx) => {
      const showDivider = dividerYear && 
        row.year >= dividerYear && 
        previousYear < dividerYear;
      
      previousYear = row.year;
  return (
    <React.Fragment key={idx}>
      {idx === 0 && (
        <tr className="bg-gray-100">
          <td colSpan={columns.length} className="px-4 py-2 text-sm font-medium text-gray-500">
            Actual Stats
          </td>
        </tr>
      )}
      {showDivider && (
        <tr className="bg-gray-100">
          <td colSpan={columns.length} className="px-4 py-2 text-sm font-medium text-gray-500">
            Projected Stats
          </td>
        </tr>
      )}
      <tr className="hover:bg-gray-50">
        {columns.map((column) => (
          <td
            key={column.key}
            className={`
              px-4 py-2 whitespace-nowrap text-sm
              ${column.align === 'right' ? 'text-right' : 
                column.align === 'center' ? 'text-center' : 'text-left'}
            `}
          >
            {column.renderCell 
              ? column.renderCell(row[column.key])
              : column.formatter && row[column.key] !== null
                ? formatters[column.formatter](row[column.key])
                : row[column.key]}
          </td>
        ))}
      </tr>
    </React.Fragment>
  );
});
};

return (
  <div className={`overflow-x-auto ${className}`}>
    <table className="min-w-full divide-y divide-gray-200">
      <thead className="bg-gray-50">
        <tr>
          {columns.map((column) => (
            <th
              key={column.key}
              onClick={() => handleSort(column.key)}
              className={`
                px-4 py-3 
                text-xs font-medium 
                text-gray-500 
                uppercase 
                tracking-wider 
                cursor-pointer
                hover:bg-gray-100
                ${column.align === 'right' ? 'text-right' : 
                  column.align === 'center' ? 'text-center' : 'text-left'}
              `}
            >
              {column.header}
              {sortColumn === column.key && (
                <span className="ml-1">
                  {sortDirection === 'asc' ? '▲' : '▼'}
                </span>
              )}
            </th>
          ))}
        </tr>
      </thead>
      <tbody className="bg-white divide-y divide-gray-200">
        {renderTableRows()}
      </tbody>
    </table>
  </div>
);
};

export default StatTable;