import { useEffect, useState } from 'react';
import { Trash2, TrendingDown, DollarSign, Percent, Loader2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area, PieChart, Pie, Cell, Legend } from 'recharts';
import { api, type WasteStatsResponse } from '@/lib/api';

const tt = {
  contentStyle: { backgroundColor: '#0f1623', border: '1px solid #f59e0b33', borderRadius: '12px', color: '#f1f5f9', fontSize: '12px' },
  itemStyle: { color: '#f1f5f9' },
};

const WasteManagementTab = () => {
  const [data, setData] = useState<WasteStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.wasteStats()
      .then(setData)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64 gap-3 text-muted-foreground">
      <Loader2 className="w-5 h-5 animate-spin" />
      <span>Loading waste data...</span>
    </div>
  );

  if (error || !data) return (
    <div className="flex items-center justify-center h-64 text-destructive text-sm">
      Failed to load data — is the backend running at localhost:8000?
    </div>
  );

  const { kpis, waste_by_meal, weekly_trend, accuracy_donut, waste_by_phase, insights } = data;

  return (
    <div className="space-y-6 animate-fade-in">
      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { icon: Trash2,       label: 'Total Estimated Waste', value: kpis.total_waste.toLocaleString(),                          sub: 'portions' },
          { icon: TrendingDown, label: 'Avg Daily Waste',       value: kpis.avg_daily_waste.toString(),                            sub: 'portions/day' },
          { icon: Percent,      label: 'Waste Rate',            value: `${kpis.waste_rate}%`,                                      sub: 'of total prepared' },
          { icon: DollarSign,   label: 'Cost of Waste',         value: `₹${kpis.cost_of_waste.toLocaleString()}`,                  sub: 'estimated loss' },
        ].map(({ icon: Icon, label, value, sub }) => (
          <div key={label} className="card-surface p-5 space-y-2">
            <div className="flex items-center gap-2">
              <Icon className="w-4 h-4 text-primary" />
              <p className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">{label}</p>
            </div>
            <p className="text-2xl font-outfit font-bold">{value}</p>
            <p className="text-xs text-muted-foreground">{sub}</p>
          </div>
        ))}
      </div>

      {/* Waste by Meal */}
      <div className="card-surface p-6">
        <h3 className="font-outfit font-bold mb-4">Waste by Meal Type</h3>
        <ResponsiveContainer width="100%" height={200}>
          <BarChart data={waste_by_meal} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
            <XAxis type="number" stroke="#64748b" fontSize={11} />
            <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={12} width={80} />
            <Tooltip {...tt} />
            <Bar dataKey="portions" fill="#f59e0b" radius={[0,8,8,0]} animationDuration={1500} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Two col charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card-surface p-6">
          <h3 className="font-outfit font-bold mb-4">Weekly Waste Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={weekly_trend}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
              <XAxis dataKey="week" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip {...tt} />
              <Area type="monotone" dataKey="waste" stroke="#f59e0b" fill="#f59e0b"
                fillOpacity={0.15} strokeWidth={2} animationDuration={1500} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card-surface p-6">
          <h3 className="font-outfit font-bold mb-4">Waste by Prediction Accuracy</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={accuracy_donut} cx="50%" cy="50%" innerRadius={60} outerRadius={90}
                dataKey="value" animationDuration={1500} paddingAngle={3}>
                {accuracy_donut.map((entry, i) => <Cell key={i} fill={entry.color} />)}
              </Pie>
              <Tooltip {...tt} />
              <Legend wrapperStyle={{ fontSize: '11px', color: '#64748b' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Waste by Phase */}
      <div className="card-surface p-6">
        <h3 className="font-outfit font-bold mb-4">Waste by Semester Phase</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={waste_by_phase}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
            <XAxis dataKey="phase" stroke="#64748b" fontSize={12} />
            <YAxis stroke="#64748b" fontSize={11} />
            <Tooltip {...tt} />
            <Bar dataKey="breakfast" name="Breakfast" fill="#f59e0b" radius={[4,4,0,0]} animationDuration={1500} />
            <Bar dataKey="lunch"     name="Lunch"     fill="#3b82f6" radius={[4,4,0,0]} animationDuration={1500} />
            <Bar dataKey="dinner"    name="Dinner"    fill="#10b981" radius={[4,4,0,0]} animationDuration={1500} />
            <Legend wrapperStyle={{ fontSize: '11px' }} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Insights */}
      <div className="card-surface p-6 border-l-4 border-l-primary">
        <h3 className="font-outfit font-bold mb-4">Waste Reduction Insights</h3>
        <ul className="space-y-3">
          {insights.map((t, i) => (
            <li key={i} className="flex items-start gap-3 text-sm text-foreground/80">
              <span className="mt-1 w-2 h-2 rounded-full bg-primary flex-shrink-0" />
              {t}
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
};

export default WasteManagementTab;
