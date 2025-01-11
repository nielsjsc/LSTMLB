import React, { useState } from 'react';
import { formatters, StatFormatterType } from '../../../utils/statFormatters';

export interface Column {
  key: string;
  header: string;
  formatter?: StatFormatterType;
  align?: 'left' | 'center' | 'right';
}

export interface StatTableProps {
  data: any[];
  columns: Column[];
  defaultSort?: string;
  className?: string;
}

// Create base table component
const StatTable: React.FC<StatTableProps> = ({
  data,
  columns,
  defaultSort,
  className = ''
}) => {
  const [sortColumn, setSortColumn] = useState(defaultSort || columns[0].key);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc');


  const handleSort = (column: string) => {
    if (sortColumn === column) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortColumn(column);
      // Change initial direction to 'asc' when new column selected
      setSortDirection('asc');
    }
  };

  const sortedData = React.useMemo(() => {
    return [...data].sort((a, b) => {
      const aValue = a[sortColumn];
      const bValue = b[sortColumn];
      if (aValue === bValue) return 0;
      if (aValue === null) return 1;
      if (bValue === null) return -1;
      return sortDirection === 'asc' 
        ? (aValue > bValue ? 1 : -1)
        : (aValue < bValue ? 1 : -1);
    });
  }, [data, sortColumn, sortDirection]);

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
                  px-4 py-2 text-xs font-medium tracking-wider text-gray-500 
                  uppercase cursor-pointer hover:bg-gray-100
                  ${column.align === 'right' ? 'text-right' : 
                    column.align === 'center' ? 'text-center' : 'text-left'}
                `}
              >
                {column.header}
                {sortColumn === column.key && (
                  <span className="ml-1">
                    {sortDirection === 'asc' ? '↑' : '↓'}
                  </span>
                )}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {sortedData.map((row, idx) => (
            <tr key={idx} className="hover:bg-gray-50">
              {columns.map((column) => (
                <td
                  key={column.key}
                  className={`
                    px-4 py-2 whitespace-nowrap text-sm
                    ${column.align === 'right' ? 'text-right' : 
                      column.align === 'center' ? 'text-center' : 'text-left'}
                  `}
                >
                  {column.formatter 
                    ? formatters[column.formatter](row[column.key])
                    : row[column.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default StatTable;