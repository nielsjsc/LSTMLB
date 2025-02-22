import React, { useState } from 'react'
import { FaChevronDown, FaChevronUp } from 'react-icons/fa'

interface CollapsibleSectionProps {
  title: string
  subtitle?: string
  icon?: React.ElementType
  children: React.ReactNode
  defaultOpen?: boolean
}

const CollapsibleSection = ({ 
  title, 
  subtitle, 
  icon: Icon, 
  children, 
  defaultOpen = false 
}: CollapsibleSectionProps) => {
  const [isOpen, setIsOpen] = useState(defaultOpen)

  return (
    <div className="border border-slate-200 dark:border-slate-700 rounded-xl overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 flex items-center justify-between bg-white/50 
                 dark:bg-slate-800/50 hover:bg-slate-50 dark:hover:bg-slate-700/50 
                 transition-colors"
      >
        <div className="flex items-center gap-3">
          {Icon && <Icon className="text-emerald-500 text-xl" />}
          <div className="text-left">
            <h3 className="text-lg font-semibold text-slate-900 dark:text-white">
              {title}
            </h3>
            {subtitle && (
              <p className="text-sm text-slate-500 dark:text-slate-400">
                {subtitle}
              </p>
            )}
          </div>
        </div>
        {isOpen ? (
          <FaChevronUp className="text-slate-400" />
        ) : (
          <FaChevronDown className="text-slate-400" />
        )}
      </button>
      
      {isOpen && (
        <div className="p-6 bg-white/30 dark:bg-slate-800/30">
          {children}
        </div>
      )}
    </div>
  )
}

export default CollapsibleSection