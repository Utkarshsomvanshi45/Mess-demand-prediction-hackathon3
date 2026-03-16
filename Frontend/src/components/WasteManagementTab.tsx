import { Trash2, TrendingDown, DollarSign, Percent } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Area, AreaChart, PieChart, Pie, Cell, Legend } from 'recharts';

const WASTE_BY_MEAL = [
  { name: 'Breakfast', portions: 680, pct: 24 },
  { name: 'Lunch', portions: 1240, pct: 43.7 },
  { name: 'Dinner', portions: 920, pct: 32.4 },
];

const WEEKLY_TREND = [
  { week: 'W1', waste: 52 }, { week: 'W2', waste: 45 }, { week: 'W3', waste: 38 },
  { week: 'W4', waste: 41 }, { week: 'W5', waste: 36 }, { week: 'W6', waste: 44 },
  { week: 'W7', waste: 39 }, { week: 'W8', waste: 35 },
];

const ACCURACY_DONUT = [
  { name: 'Correctly predicted Low\n(waste avoided)', value: 45, color: '#10b981' },
  { name: 'Predicted Medium but was Low\n(moderate waste)', value: 35, color: '#f59e0b' },
  { name: 'Predicted High but was Low\n(maximum waste)', value: 20, color: '#ef4444' },
];

const WASTE_BY_PHASE = [
  { phase: 'Regular', breakfast: 620, lunch: 780, dinner: 400 },
  { phase: 'Exams', breakfast: 180, lunch: 280, dinner: 180 },
  { phase: 'Holidays', breakfast: 120, lunch: 160, dinner: 120 },
];

const chartTooltipStyle = {
  contentStyle: { backgroundColor: '#0f1623', border: '1px solid #f59e0b33', borderRadius: '12px', color: '#f1f5f9', fontSize: '12px' },
  itemStyle: { color: '#f1f5f9' },
};

const WasteManagementTab = () => {
  return (
    <div className="space-y-6 animate-fade-in">
      {/* KPIs */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { icon: Trash2, label: 'Total Estimated Waste', value: '2,840', sub: 'portions' },
          { icon: TrendingDown, label: 'Avg Daily Waste', value: '41', sub: 'portions/day' },
          { icon: Percent, label: 'Waste Rate', value: '18.3%', sub: 'of total prepared' },
          { icon: DollarSign, label: 'Cost of Waste', value: '₹1,27,800', sub: 'estimated loss' },
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
          <BarChart data={WASTE_BY_MEAL} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
            <XAxis type="number" stroke="#64748b" fontSize={11} />
            <YAxis type="category" dataKey="name" stroke="#64748b" fontSize={12} width={80} />
            <Tooltip {...chartTooltipStyle} />
            <Bar dataKey="portions" fill="#f59e0b" radius={[0, 8, 8, 0]} animationDuration={1500} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Two col charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card-surface p-6">
          <h3 className="font-outfit font-bold mb-4">Weekly Waste Trend</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={WEEKLY_TREND}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
              <XAxis dataKey="week" stroke="#64748b" fontSize={11} />
              <YAxis stroke="#64748b" fontSize={11} />
              <Tooltip {...chartTooltipStyle} />
              <Area type="monotone" dataKey="waste" stroke="#f59e0b" fill="#f59e0b" fillOpacity={0.15} strokeWidth={2} animationDuration={1500} />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="card-surface p-6">
          <h3 className="font-outfit font-bold mb-4">Waste by Prediction Accuracy</h3>
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie data={ACCURACY_DONUT} cx="50%" cy="50%" innerRadius={60} outerRadius={90} dataKey="value" animationDuration={1500} paddingAngle={3}>
                {ACCURACY_DONUT.map((entry, i) => <Cell key={i} fill={entry.color} />)}
              </Pie>
              <Tooltip {...chartTooltipStyle} />
              <Legend wrapperStyle={{ fontSize: '11px', color: '#64748b' }} />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Phase */}
      <div className="card-surface p-6">
        <h3 className="font-outfit font-bold mb-4">Waste by Semester Phase</h3>
        <ResponsiveContainer width="100%" height={250}>
          <BarChart data={WASTE_BY_PHASE}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
            <XAxis dataKey="phase" stroke="#64748b" fontSize={12} />
            <YAxis stroke="#64748b" fontSize={11} />
            <Tooltip {...chartTooltipStyle} />
            <Bar dataKey="breakfast" name="Breakfast" fill="#f59e0b" radius={[4, 4, 0, 0]} animationDuration={1500} />
            <Bar dataKey="lunch" name="Lunch" fill="#3b82f6" radius={[4, 4, 0, 0]} animationDuration={1500} />
            <Bar dataKey="dinner" name="Dinner" fill="#10b981" radius={[4, 4, 0, 0]} animationDuration={1500} />
            <Legend wrapperStyle={{ fontSize: '11px' }} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Insights */}
      <div className="card-surface p-6 border-l-4 border-l-primary">
        <h3 className="font-outfit font-bold mb-4">Waste Reduction Insights</h3>
        <ul className="space-y-3">
          {[
            'Sunday dinners have 34% higher waste due to low occupancy — consider reducing preparation by 30%',
            'Holiday breakfast waste peaks at 52% — switch to low-demand menu items during holidays',
            'Dessert days show 22% lower waste — dessert increases consumption across all demand levels',
          ].map((t, i) => (
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
