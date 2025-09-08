import { QueryClientProvider } from '@tanstack/react-query';
import { Switch, Route, useLocation, Router } from 'wouter';
import TempLineageV1 from '@/pages/temp-lineage-v1'; // TEMPORARY
import LegacyLineageAdapter from '@/adapters/LegacyLineageAdapter'; // Direct import adapter
import { ThemeProvider } from '@/components/theme-provider';
import { Footer } from '@/components/ui/footer';
import { Header } from '@/components/ui/header';
import { Toaster } from '@/components/ui/toaster';
import { TooltipProvider } from '@/components/ui/tooltip';
import Home from "@/pages/home";
import HomeChat from "@/pages/home-chat";
import Explore from "@/pages/explore";
import Lineage from '@/pages/lineage';
import TransformationLineage from '@/pages/transformation-lineage';
import Login from '@/pages/login';
import NotFound from '@/pages/not-found';
import { useAuthStore } from '@/store/authStore';
import { queryClient } from '@/utils/queryClient';
import { ChatProvider } from '@/components/chat';

function AppContent() {
  const [location] = useLocation();
  const currentPage =
    location === '/'
      ? 'home'
      : location === '/home-chat'
        ? 'home-chat'
      : location === '/explore'
        ? 'explore'
        : location === '/lineage'
          ? 'lineage'
          : 'home';
  const isAuthenticated = useAuthStore(state => state.isAuthenticated);

  // If not authenticated, show login page
  if (!isAuthenticated) {
    return <Login />;
  }

  // Special layout for explore and lineage pages with full-height sidebar
  if (location === '/explore' || location === '/lineage') {
    return (
      <div className="min-h-screen bg-gray-100 dark:bg-gray-900 transition-colors flex flex-col">
        <Header currentPage={currentPage} />
        <div className="flex-1 relative">{location === '/explore' ? <Explore /> : <Lineage />}</div>
        <Footer />
        <Toaster />
      </div>
    );
  }


  return (
    <div className="min-h-screen bg-gray-100 dark:bg-gray-900 transition-colors">
      <Header currentPage={currentPage} />
      <div className="pt-16">
        <Switch>
          <Route path="/" component={Home} />
          <Route path="/home-chat" component={HomeChat} />
          <Route path="/transformation-lineage" component={TransformationLineage} />
          <Route path="/lineage-v1" component={TempLineageV1} /> {/* TEMPORARY ROUTE */}
          <Route path="/lineage-adapter" component={LegacyLineageAdapter} />{' '}
          {/* Direct import adapter */}
          <Route component={NotFound} />
        </Switch>
      </div>
      <Footer />
      <Toaster />
    </div>
  );
}

function App() {
  // Get the base path from the environment variable or default to '/'
  const basePath = import.meta.env.VITE_BASE_PATH || '/';
  
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider defaultTheme="dark" storageKey="dataharbor-ui-theme">
        <TooltipProvider>
          <ChatProvider>
            <Router base={basePath}>
              <AppContent />
            </Router>
          </ChatProvider>
        </TooltipProvider>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export default App;
