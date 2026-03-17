import { useEffect, useState } from 'react';
import { Database, Columns, Target, Info, Loader2 } from 'lucide-react';
import { api, type DataOverviewResponse } from '@/lib/api';

const demandColor = (val: string) => {
  if (val === 'High')   return 'text-demand-high font-semibold';
  if (val === 'Medium') return 'text-demand-medium font-semibold';
  if (val === 'Low')    return 'text-demand-low font-semibold';
  return '';
};

const DataOverviewTab = () => {
  const [data, setData] = useState<DataOverviewResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.dataOverview()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3 text-muted-foreground">
      <Loader2 className="w-5 h-5 animate-spin" />
      <span>Loading dataset...</span>
    </div>
  );

  if (error || !data) return (
    <div className="flex items-center justify-center h-64 text-destructive text-sm">
      Failed to load data — is the backend running at localhost:8000?
    </div>
  );

  const columns = data.preview.length > 0 ? Object.keys(data.preview[0]) : [];

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Stat cards */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { icon: Database, label: 'Total Records',  value: data.total_records.toLocaleString(), accent: false },
          { icon: Columns,  label: 'Input Features', value: data.total_features.toString(),       accent: false },
          { icon: Target,   label: 'Target Column',  value: data.target_column,                   accent: true },
        ].map(({ icon: Icon, label, value, accent }) => (
          <div key={label} className="card-surface p-5 space-y-2">
            <div className="flex items-center gap-2">
              <Icon className="w-4 h-4 text-primary" />
              <p className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">{label}</p>
            </div>
            <p className={`text-xl font-outfit font-bold ${accent ? 'text-success' : ''}`}>{value}</p>
          </div>
        ))}
      </div>

      {/* Dataset Preview */}
      <div className="card-surface p-6 overflow-x-auto">
        <h3 className="font-outfit font-bold mb-4">Dataset Preview</h3>
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border">
              {columns.map((col) => (
                <th key={col} className="px-3 py-2 text-left text-muted-foreground font-medium whitespace-nowrap">
                  {col}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.preview.map((row, i) => (
              <tr key={i} className={`border-b border-border/30 ${i % 2 === 1 ? 'bg-background/30' : ''}`}>
                {columns.map((key) => (
                  <td key={key} className={`px-3 py-2 whitespace-nowrap ${key === 'demand_level' ? demandColor(String(row[key])) : ''}`}>
                    {String(row[key])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary Stats */}
      <div className="card-surface p-6">
        <h3 className="font-outfit font-bold mb-4">Summary Statistics</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              {['Column', 'Mean', 'Min', 'Max', 'Std Dev'].map((h) => (
                <th key={h} className="px-4 py-2 text-left text-muted-foreground font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.summary_stats.map((s) => (
              <tr key={s.column} className="border-b border-border/30">
                <td className="px-4 py-2 font-mono text-primary text-xs">{s.column}</td>
                <td className="px-4 py-2">{s.mean}</td>
                <td className="px-4 py-2">{s.min}</td>
                <td className="px-4 py-2">{s.max}</td>
                <td className="px-4 py-2">{s.std}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Data Source Note */}
      <div className="card-surface p-5 flex items-start gap-3">
        <Info className="w-5 h-5 text-primary flex-shrink-0 mt-0.5" />
        <div className="space-y-1">
          <p className="text-sm font-medium">Data Source Note</p>
          <p className="text-xs text-muted-foreground leading-relaxed">
            This dataset is synthetically generated to simulate real-world university mess operations.
            It includes meal types, primary dishes, occupancy rates, menu add-ons, and demand levels
            across different semester phases and days of the week.
          </p>
        </div>
      </div>
    </div>
  );
};

export default DataOverviewTab;
