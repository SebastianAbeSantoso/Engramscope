import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'

// GitHub Pages serves the app under /Engramscope/.
export default defineConfig({
  base: process.env.ENGRAMSCOPE_BASE ?? '/Engramscope/',
  plugins: [react()],
  worker: { format: 'es' },
  test: { environment: 'node' },
})
