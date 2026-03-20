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
    <div className="border border-gray-200 rounded-xl overflow-hidden">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-6 py-4 flex items-center justify-between 
                 bg-white hover:bg-white 
                 transition-colors"
      >
        <div className="flex items-center gap-3">
          {Icon && <Icon className="text-brand-500 text-xl" />}
          <div className="text-left">
            <h3 className="text-lg font-semibold text-gray-900">
              {title}
            </h3>
            {subtitle && (
              <p className="text-sm text-gray-500">
                {subtitle}
              </p>
            )}
          </div>
        </div>
        {isOpen ? (
          <FaChevronUp className="text-gray-500" />
        ) : (
          <FaChevronDown className="text-gray-500" />
        )}
      </button>
      
      {isOpen && (
        <div className="p-6 bg-gray-50">
          {children}
        </div>
      )}
    </div>
  )
}

export default CollapsibleSection