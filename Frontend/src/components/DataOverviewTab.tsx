import { Database, Columns, Target, Info } from 'lucide-react';
import { MOCK_DATASET, type DemandLevel } from '@/lib/constants';

const demandClass = (d: DemandLevel) =>
  d === 'High' ? 'text-demand-high' : d === 'Medium' ? 'text-demand-medium' : 'text-demand-low';

const SUMMARY_STATS = [
  { col: 'hostel_occupancy_pct', mean: 74.2, min: 32, max: 98, std: 16.8 },
  { col: 'has_paneer', mean: 0.18, min: 0, max: 1, std: 0.38 },
  { col: 'has_chicken', mean: 0.12, min: 0, max: 1, std: 0.32 },
  { col: 'has_egg', mean: 0.15, min: 0, max: 1, std: 0.36 },
  { col: 'has_dessert', mean: 0.34, min: 0, max: 1, std: 0.47 },
];

const DataOverviewTab = () => {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* Stat cards */}
      <div className="grid grid-cols-3 gap-4">
        {[
          { icon: Database, label: 'Total Records', value: '1,020' },
          { icon: Columns, label: 'Input Features', value: '12' },
          { icon: Target, label: 'Target Column', value: 'demand_level', accent: true },
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
              {Object.keys(MOCK_DATASET[0]).map((col) => (
                <th key={col} className="px-3 py-2 text-left text-muted-foreground font-medium whitespace-nowrap">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MOCK_DATASET.map((row, i) => (
              <tr key={i} className={`border-b border-border/30 ${i % 2 === 1 ? 'bg-background/30' : ''}`}>
                {Object.entries(row).map(([key, val]) => (
                  <td key={key} className={`px-3 py-2 whitespace-nowrap ${key === 'demand_level' ? demandClass(val as DemandLevel) + ' font-semibold' : ''}`}>
                    {String(val)}
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
            {SUMMARY_STATS.map((s) => (
              <tr key={s.col} className="border-b border-border/30">
                <td className="px-4 py-2 font-mono text-primary text-xs">{s.col}</td>
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
            This dataset is synthetically generated to simulate real-world university mess operations. It includes meal types, primary dishes, occupancy rates, menu add-ons, and demand levels across different semester phases and days of the week.
          </p>
        </div>
      </div>
    </div>
  );
};

export default DataOverviewTab;
