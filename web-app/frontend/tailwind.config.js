/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        sans: ['Inter', '-apple-system', 'BlinkMacSystemFont', 'Segoe UI', 'Roboto', 'sans-serif'],
        display: ['Barlow', 'Inter', '-apple-system', 'sans-serif'],
        mono: ['JetBrains Mono', 'Menlo', 'Consolas', 'monospace'],
      },
      colors: {
        brand: {
          50: '#fef2f3',
          100: '#fde6e8',
          200: '#f9c0c6',
          300: '#f28b97',
          400: '#E8384F',
          500: '#C41E3A',   // Primary accent — classic baseball red
          600: '#A3162E',
          700: '#7F1122',
          800: '#6B1020',
          900: '#3B0D12',
        },
        navy: {
          50: '#E8EDF5',
          100: '#C5D1E8',
          200: '#8FA6CC',
          300: '#5A7BB0',
          400: '#2E5694',
          500: '#1A3F6F',
          600: '#0F2D4D',
          700: '#0C2340',   // MLB official navy
          800: '#091B30',
          900: '#060F1C',
        },
        accent: {
          blue: '#1E5CA6',
          indigo: '#6366F1',
          sky: '#0284C7',
        },
        mlb: {
          blue: '#002D72',
          red: '#E31937',
          navy: '#0C2340',
        },
      },
      boxShadow: {
        'glow': '0 1px 3px rgba(0, 0, 0, 0.08)',
        'glow-lg': '0 4px 12px rgba(0, 0, 0, 0.1)',
        'inner-glow': 'inset 0 1px 0 rgba(255,255,255,0.6)',
        'card': '0 1px 3px rgba(0, 0, 0, 0.06), 0 1px 2px rgba(0, 0, 0, 0.04)',
        'card-hover': '0 4px 12px rgba(0, 0, 0, 0.08)',
      },
      backgroundImage: {
        'gradient-brand': 'linear-gradient(135deg, #C41E3A, #A3162E)',
        'gradient-navy': 'linear-gradient(135deg, #0C2340, #163A5F)',
      }
    },
  },
  plugins: [],
}