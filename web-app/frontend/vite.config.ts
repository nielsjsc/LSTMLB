import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// Detect Vercel environment
const isVercel = process.env.VERCEL === '1'

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'vercel-path-resolution',
      configResolved(config) {
        console.log('Build environment:', {
          isVercel,
          root: config.root,
          base: config.base
        })
      }
    }
  ],
  base: isVercel ? '/' : './',
  build: {
    outDir: 'dist',
    sourcemap: true,
    rollupOptions: {
      input: {
        main: path.resolve(__dirname, 'index.html')
      }
    }
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  }
})