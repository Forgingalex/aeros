/**
 * AEROS Design System
 * Futuristic aerospace autonomy software design tokens
 */

export const theme = {
  colors: {
    primary: '#0D1B61',
    accent: '#1C7DF2',
    background: '#0A0F1C',
    surface: '#13182B',
    surfaceElevated: '#1A2138',
    
    status: {
      connected: '#00D67F',
      warning: '#F6C744',
      error: '#E03232',
    },
    
    text: {
      primary: '#FFFFFF',
      secondary: 'rgba(255, 255, 255, 0.7)',
      muted: 'rgba(255, 255, 255, 0.5)',
      disabled: 'rgba(255, 255, 255, 0.3)',
    },
    
    border: {
      default: 'rgba(255, 255, 255, 0.1)',
      hover: 'rgba(255, 255, 255, 0.15)',
      active: 'rgba(28, 125, 242, 0.3)',
    },
  },
  
  typography: {
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Segoe UI', sans-serif",
    fontFamilyMono: "'SF Mono', 'Monaco', 'Menlo', 'Consolas', monospace",
    
    scale: {
      xs: '12px',
      sm: '14px',
      base: '16px',
      lg: '20px',
      xl: '28px',
      '2xl': '36px',
    },
    
    weight: {
      regular: 400,
      medium: 500,
      semibold: 600,
      bold: 700,
    },
  },
  
  spacing: {
    xs: '0.5rem',
    sm: '0.75rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
    '2xl': '3rem',
  },
  
  borderRadius: {
    sm: '8px',
    md: '12px',
    lg: '16px',
    xl: '20px',
    full: '9999px',
  },
  
  shadows: {
    sm: '0 1px 2px rgba(0, 0, 0, 0.1)',
    md: '0 4px 8px rgba(0, 0, 0, 0.15)',
    lg: '0 8px 16px rgba(0, 0, 0, 0.2)',
    none: 'none',
  },
  
  transitions: {
    fast: '150ms ease',
    base: '200ms ease',
    slow: '300ms ease',
  },
  
  zIndex: {
    base: 1,
    elevated: 10,
    overlay: 100,
    modal: 1000,
  },
};

export default theme;

