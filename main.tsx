import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';
import { setDebugMode, isDebugMode } from './utils/logger';

// Make debug utilities available in browser console
(window as any).dataharborDebug = {
  enable: () => {
    setDebugMode(true);
    console.log('Debug mode enabled. Refresh the page to see debug logs.');
  },
  disable: () => {
    setDebugMode(false);
    console.log('Debug mode disabled. Refresh the page to hide debug logs.');
  },
  status: () => {
    console.log(`Debug mode is ${isDebugMode() ? 'enabled' : 'disabled'}`);
  },
  // Add debug utilities for field connections
  showFieldConnections: () => {
    const reactFlowInstance = document.querySelector('.react-flow')?.getElementsByClassName('react-flow__edges')[0];
    if (reactFlowInstance) {
      const edges = Array.from(reactFlowInstance.querySelectorAll('[data-field-connection="true"]'));
      console.log('Field connection edges in DOM:', edges.length);
      edges.forEach(edge => console.log('Edge:', edge));
    }
  }
};

// Log initial status
if (isDebugMode()) {
  console.log('%c🚀 DataHarbor Debug Mode Active', 'background: #9333ea; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold;');
  console.log('Use dataharborDebug.disable() to turn off debug logs');
}

createRoot(document.getElementById('root')!).render(<App />);
