import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'log-build',
      configResolved(config) {
        console.log('Build Config:', {
          root: config.root,
          base: config.base,
          publicDir: config.publicDir,
          build: {
            outDir: config.build.outDir,
            assetsDir: config.build.assetsDir,
            rollupOptions: config.build.rollupOptions
          }
        })
      },
      buildStart() {
        console.log('Build starting, entry point resolution...')
      },
      resolveId(source) {
        if (source.includes('main.tsx')) {
          console.log('Resolving main.tsx from:', source)
        }
        return null
      }
    }
  ],
  build: {
    outDir: 'dist',
    sourcemap: true
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src')
    }
  }
})