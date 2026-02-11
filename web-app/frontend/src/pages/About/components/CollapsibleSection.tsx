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
    <div className="border border-white/[0.06] rounded-xl overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 flex items-center justify-between 
                 bg-surface-800/50 hover:bg-surface-700/50 
                 transition-colors"
      >
        <div className="flex items-center gap-3">
          {Icon && <Icon className="text-brand-400 text-xl" />}
          <div className="text-left">
            <h3 className="text-lg font-semibold text-white">
              {title}
            </h3>
            {subtitle && (
              <p className="text-sm text-surface-400">
                {subtitle}
              </p>
            )}
          </div>
        </div>
        {isOpen ? (
          <FaChevronUp className="text-surface-400" />
        ) : (
          <FaChevronDown className="text-surface-400" />
        )}
      </button>
      
      {isOpen && (
        <div className="p-6 bg-surface-800/30">
          {children}
        </div>
      )}
    </div>
  )
}

export default CollapsibleSection