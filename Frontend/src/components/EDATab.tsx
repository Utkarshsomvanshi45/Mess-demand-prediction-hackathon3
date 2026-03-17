import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, Legend } from 'recharts';
import { Database, BarChart3, Activity, Calendar, Loader2 } from 'lucide-react';
import { api, type EDAResponse } from '@/lib/api';

const tt = {
  contentStyle: { backgroundColor: '#0f1623', border: '1px solid #f59e0b33', borderRadius: '12px', color: '#f1f5f9', fontSize: '12px' },
};

const EDATab = () => {
  const [data, setData] = useState<EDAResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.eda()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3 text-muted-foreground">
      <Loader2 className="w-5 h-5 animate-spin" />
      <span>Loading EDA data...</span>
    </div>
  );

  if (error || !data) return (
    <div className="flex items-center justify-center h-64 text-destructive text-sm">
      Failed to load data — is the backend running at localhost:8000?
    </div>
  );

  const { kpis, demand_dist, demand_by_meal, occupancy_box, phase_demand, day_demand } = data;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { icon: Database,  label: 'Total Records',     value: kpis.total_records.toLocaleString() },
          { icon: Activity,  label: 'Avg Occupancy',     value: `${kpis.avg_occupancy}%` },
          { icon: BarChart3, label: 'High Demand Count', value: kpis.high_demand_count.toLocaleString() },
          { icon: Calendar,  label: 'Weekend Records',   value: kpis.weekend_records.toLocaleString() },
        ].map(({ icon: Icon, label, value }) => (
          <div key={label} className="card-surface p-5 space-y-2">
            <div className="flex items-center gap-2">
              <Icon className="w-4 h-4 text-primary" />
              <p className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">{label}</p>
            </div>
            <p className="text-2xl font-outfit font-bold">{value}</p>
          </div>
        ))}
      </div>

      {/* 2x2 grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Donut */}
        <div className="card-surface p-6">
          <h3 className="font-outfit font-bold mb-4">Demand Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={demand_dist} cx="50%" cy="50%" innerRadius={55} outerRadius={85}
                dataKey="value" animationDuration={1500} paddingAngle={3}>
                {demand_dist.map((e, i) => <Cell key={i} fill={e.color} />)}
              </Pie>
              <Tooltip {...tt} />
              <Legend wrapperStyle={{ fontSize: '12px' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* Grouped bar */}
        <div className="card-surface p-6">
          <h3 className="font-outfit font-bold mb-4">Demand by Meal Type</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={demand_by_meal}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
              <XAxis dataKey="meal" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip {...tt} />
              <Bar dataKey="High"   fill="#ef4444" radius={[4,4,0,0]} animationDuration={1500} />
              <Bar dataKey="Medium" fill="#f59e0b" radius={[4,4,0,0]} animationDuration={1500} />
              <Bar dataKey="Low"    fill="#10b981" radius={[4,4,0,0]} animationDuration={1500} />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Box plot */}
        <div className="card-surface p-6">
          <h3 className="font-outfit font-bold mb-4">Hostel Occupancy vs Demand</h3>
          <div className="space-y-4 pt-2">
            {occupancy_box.map((item) => {
              const color = item.level === 'High' ? '#ef4444' : item.level === 'Medium' ? '#f59e0b' : '#10b981';
              const rangeWidth = ((item.max - item.min) / 100) * 100;
              const leftOffset = ((item.min - 20) / 100) * 100;
              const iqrLeft    = ((item.q1 - item.min) / (item.max - item.min)) * 100;
              const iqrWidth   = ((item.q3 - item.q1) / (item.max - item.min)) * 100;
              const medianPos  = ((item.median - item.min) / (item.max - item.min)) * 100;
              return (
                <div key={item.level} className="flex items-center gap-4">
                  <span className="w-16 text-xs font-medium" style={{ color }}>{item.level}</span>
                  <div className="flex-1 relative h-8">
                    <div className="absolute top-1/2 -translate-y-1/2 h-px bg-muted-foreground/40"
                      style={{ left: `${leftOffset}%`, width: `${rangeWidth}%` }} />
                    <div className="absolute top-1 h-6 rounded"
                      style={{ left: `${leftOffset + (iqrLeft / 100) * rangeWidth}%`, width: `${(iqrWidth / 100) * rangeWidth}%`, backgroundColor: `${color}22`, border: `1px solid ${color}` }} />
                    <div className="absolute top-0.5 w-0.5 h-7"
                      style={{ left: `${leftOffset + (medianPos / 100) * rangeWidth}%`, backgroundColor: color }} />
                    <span className="absolute -bottom-4 text-[9px] text-muted-foreground" style={{ left: `${leftOffset}%` }}>{item.min}%</span>
                    <span className="absolute -bottom-4 text-[9px] text-muted-foreground" style={{ left: `${leftOffset + rangeWidth}%`, transform: 'translateX(-100%)' }}>{item.max}%</span>
                  </div>
                </div>
              );
            })}
          </div>
          <div className="flex gap-4 mt-8 text-[10px] text-muted-foreground">
            <span>Box = IQR (Q1–Q3)</span><span>Line = Median</span><span>Whiskers = Min–Max</span>
          </div>
        </div>

        {/* Stacked bar */}
        <div className="card-surface p-6">
          <h3 className="font-outfit font-bold mb-4">Semester Phase vs Demand</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={phase_demand}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
              <XAxis dataKey="phase" stroke="#64748b" fontSize={12} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip {...tt} />
              <Bar dataKey="High"   stackId="a" fill="#ef4444" animationDuration={1500} />
              <Bar dataKey="Medium" stackId="a" fill="#f59e0b" animationDuration={1500} />
              <Bar dataKey="Low"    stackId="a" fill="#10b981" radius={[4,4,0,0]} animationDuration={1500} />
              <Legend wrapperStyle={{ fontSize: '11px' }} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Day of Week */}
      <div className="card-surface p-6">
        <h3 className="font-outfit font-bold mb-4">Demand by Day of Week</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={day_demand}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
            <XAxis dataKey="day" stroke="#64748b" fontSize={12} />
            <YAxis stroke="#64748b" fontSize={11} />
            <Tooltip {...tt} />
            <Bar dataKey="High"   fill="#ef4444" radius={[4,4,0,0]} animationDuration={1500} />
            <Bar dataKey="Medium" fill="#f59e0b" radius={[4,4,0,0]} animationDuration={1500} />
            <Bar dataKey="Low"    fill="#10b981" radius={[4,4,0,0]} animationDuration={1500} />
            <Legend wrapperStyle={{ fontSize: '11px' }} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

export default EDATab;
