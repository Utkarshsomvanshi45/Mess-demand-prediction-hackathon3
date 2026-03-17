import { useEffect, useState } from 'react';
import { UtensilsCrossed } from 'lucide-react';
import { api, type PipelineStatusResponse } from '@/lib/api';

const DashboardHeader = () => {
  const [status, setStatus] = useState<PipelineStatusResponse | null>(null);

  useEffect(() => {
    api.pipelineStatus().then(setStatus).catch(() => null);
  }, []);

  const versionLabel = status
    ? `v${status.current_version} · ${status.model_name}`
    : 'v4 · RandomForestClassifier';

  return (
    <header className="flex items-center justify-between mb-8 pb-6 border-b border-border relative overflow-hidden">
      <div className="absolute top-0 right-0 w-64 h-64 bg-primary opacity-[0.03] blur-[100px] -mr-32 -mt-32 pointer-events-none" />
      <div className="flex items-center gap-4">
        <div className="p-3 bg-primary/10 rounded-xl">
          <UtensilsCrossed className="w-8 h-8 text-primary" />
        </div>
        <div>
          <h1 className="text-2xl font-bold font-outfit tracking-tight">
            Mess Demand & Food Waste Management
          </h1>
          <p className="text-muted-foreground text-sm">
            Intelligent forecasting & waste reduction for university operations
          </p>
        </div>
      </div>
      <div className="px-4 py-1.5 bg-primary/10 border border-primary/20 rounded-full">
        <span className="text-primary font-mono text-xs font-semibold">
          {versionLabel}
        </span>
      </div>
    </header>
  );
};

export default DashboardHeader;
