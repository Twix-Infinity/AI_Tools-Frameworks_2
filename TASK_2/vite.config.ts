import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  optimizeDeps: {
    // include lucide-react so Vite pre-bundles it (fixes certain runtime issues)
    include: ['lucide-react'],
  },
});