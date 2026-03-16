import { useState } from 'react';
import { Sparkles, Trash2, Leaf } from 'lucide-react';
import {
  type MealType, type DemandLevel, type SemesterPhase,
  DISHES_BY_MEAL, DAYS_OF_WEEK, getTier,
  calculateDemand, getWasteMetrics, MEAL_FACTORS,
} from '@/lib/constants';

const KpiCard = ({ label, value, sub }: { label: string; value: string; sub?: string }) => (
  <div className="card-surface p-4 space-y-1">
    <p className="text-xs text-muted-foreground uppercase tracking-wider">{label}</p>
    <p className="text-lg font-outfit font-bold text-foreground">{value}</p>
    {sub && <p className="text-xs text-muted-foreground">{sub}</p>}
  </div>
);

const DemandPredictionTab = () => {
  const [meal, setMeal] = useState<MealType>('Lunch');
  const [day, setDay] = useState('Monday');
  const [dish, setDish] = useState(DISHES_BY_MEAL['Lunch'][0]);
  const [phase, setPhase] = useState<SemesterPhase>('Regular');
  const [occupancy, setOccupancy] = useState(80);
  const [dessert, setDessert] = useState(false);
  const [fruit, setFruit] = useState(false);
  const [drink, setDrink] = useState(false);
  const [result, setResult] = useState<{ demand: DemandLevel; waste: ReturnType<typeof getWasteMetrics> } | null>(null);

  const tier = getTier(dish);

  const handleMealChange = (newMeal: MealType) => {
    setMeal(newMeal);
    setDish(DISHES_BY_MEAL[newMeal][0]);
    setResult(null);
  };

  const predict = () => {
    const demand = calculateDemand(occupancy, tier);
    const waste = getWasteMetrics(occupancy, meal, demand);
    setResult({ demand, waste });
  };

  const tierClass = tier === 'High' ? 'badge-tier-high' : tier === 'Medium' ? 'badge-tier-medium' : 'badge-tier-low';
  const demandColor = (d: DemandLevel) => d === 'High' ? 'text-demand-high' : d === 'Medium' ? 'text-demand-medium' : 'text-demand-low';

  const wasteColor = (pct: number) => pct > 20 ? 'text-demand-high' : pct > 10 ? 'text-demand-medium' : 'text-demand-low';
  const wasteBg = (pct: number) => pct > 20 ? 'bg-danger' : pct > 10 ? 'bg-warning' : 'bg-success';

  const tips: Record<DemandLevel, string> = {
    High: 'Expect heavy footfall. Prepare maximum portions. Consider extra staff on duty.',
    Medium: 'Moderate turnout expected. Prepare standard quantity.',
    Low: 'Light attendance expected. Reduce preparation to minimise waste.',
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[58fr_42fr] gap-6 animate-fade-in">
      {/* Left — Input */}
      <div className="card-surface p-6 space-y-5">
        <div className="flex items-center gap-2 mb-2">
          <Sparkles className="w-5 h-5 text-primary" />
          <h2 className="font-outfit font-bold text-lg">Input Parameters</h2>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <SelectField label="Meal Type" value={meal} options={['Breakfast', 'Lunch', 'Dinner']} onChange={(v) => handleMealChange(v as MealType)} />
          <SelectField label="Day of Week" value={day} options={DAYS_OF_WEEK} onChange={setDay} />
        </div>

        <SelectField label="Primary Dish" value={dish} options={DISHES_BY_MEAL[meal]} onChange={(v) => { setDish(v); setResult(null); }} />

        <div className="flex items-center gap-2">
          <span className="text-xs text-muted-foreground">Menu Tier:</span>
          <span className={tierClass}>{tier}</span>
        </div>

        <div className="grid grid-cols-2 gap-4">
          <SelectField label="Semester Phase" value={phase} options={['Regular', 'Exams', 'Holidays']} onChange={(v) => setPhase(v as SemesterPhase)} />
          <div className="space-y-2">
            <label className="text-xs text-muted-foreground">Hostel Occupancy</label>
            <div className="flex items-center gap-3">
              <input type="range" min={30} max={100} value={occupancy} onChange={(e) => { setOccupancy(+e.target.value); setResult(null); }}
                className="flex-1 accent-primary h-2 bg-border rounded-full appearance-none [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:w-4 [&::-webkit-slider-thumb]:h-4 [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-primary [&::-webkit-slider-thumb]:cursor-pointer" />
              <span className="text-primary font-semibold text-sm min-w-[3rem] text-right">{occupancy}%</span>
            </div>
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-xs text-muted-foreground">Menu Add-ons</label>
          <div className="flex gap-4">
            <Toggle label="Dessert Served" checked={dessert} onChange={setDessert} />
            <Toggle label="Fruit Served" checked={fruit} onChange={setFruit} />
            <Toggle label="Drink Served" checked={drink} onChange={setDrink} />
          </div>
        </div>

        <button onClick={predict} className="btn-primary text-sm">
          ⚡ Predict Demand
        </button>
      </div>

      {/* Right — Results */}
      <div className="space-y-4">
        {!result ? (
          <div className="card-surface border-dashed p-12 flex flex-col items-center justify-center text-center min-h-[400px]">
            <Sparkles className="w-10 h-10 text-muted-foreground mb-4" />
            <p className="text-muted-foreground text-sm">Configure parameters and click Predict Demand</p>
          </div>
        ) : (
          <>
            {/* Card A — Demand */}
            <div className="card-surface p-6 animate-fade-in space-y-3">
              <p className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">Predicted Demand</p>
              <p className={`text-5xl font-outfit font-bold ${demandColor(result.demand)}`}>{result.demand}</p>
              <p className="text-sm text-muted-foreground">{tips[result.demand]}</p>
            </div>

            {/* Card B — Waste Estimation */}
            <div className="card-surface p-6 animate-fade-in space-y-4">
              <div className="flex items-center gap-2">
                <Trash2 className="w-4 h-4 text-muted-foreground" />
                <Leaf className="w-4 h-4 text-success" />
                <p className="text-xs font-bold text-muted-foreground tracking-widest uppercase">Estimated Food Waste Impact</p>
              </div>

              <div className="grid grid-cols-3 gap-3">
                <KpiCard label="Recommended" value={`${result.waste.recommended}`} sub="portions" />
                <KpiCard label="Expected" value={`${result.waste.expected}`} sub="consumption" />
                <KpiCard label="Est. Waste" value={`${result.waste.waste}`} sub={`${result.waste.wastePct.toFixed(1)}%`} />
              </div>

              {/* Waste bar */}
              <div className="space-y-1">
                <div className="h-2 w-full bg-border rounded-full overflow-hidden">
                  <div className={`h-full ${wasteBg(result.waste.wastePct)} rounded-full transition-all duration-700`} style={{ width: `${Math.min(result.waste.wastePct, 100)}%` }} />
                </div>
                <p className={`text-xs font-semibold ${wasteColor(result.waste.wastePct)}`}>{result.waste.wastePct.toFixed(1)}% waste</p>
              </div>

              <p className="text-sm text-muted-foreground">Estimated waste cost: <span className="text-primary font-semibold">₹{result.waste.cost.toLocaleString()}</span></p>
              <p className="text-xs text-muted-foreground">Preparing {result.waste.waste} fewer portions could save ₹{result.waste.cost.toLocaleString()} today</p>
            </div>

            {/* Input Summary */}
            <div className="card-surface p-4 animate-fade-in">
              <p className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase mb-3">Input Summary</p>
              <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-xs">
                {[
                  ['Meal', meal], ['Day', day], ['Dish', dish], ['Tier', tier],
                  ['Phase', phase], ['Occupancy', `${occupancy}%`],
                  ['Dessert', dessert ? 'Yes' : 'No'], ['Fruit', fruit ? 'Yes' : 'No'], ['Drink', drink ? 'Yes' : 'No'],
                ].map(([k, v]) => (
                  <div key={k} className="flex justify-between py-1 border-b border-border/50">
                    <span className="text-muted-foreground">{k}</span>
                    <span className="text-foreground font-medium">{v}</span>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
};

const SelectField = ({ label, value, options, onChange }: { label: string; value: string; options: string[]; onChange: (v: string) => void }) => (
  <div className="space-y-2">
    <label className="text-xs text-muted-foreground">{label}</label>
    <select value={value} onChange={(e) => onChange(e.target.value)}
      className="w-full bg-background border border-border rounded-xl px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary appearance-none cursor-pointer">
      {options.map((o) => <option key={o} value={o}>{o}</option>)}
    </select>
  </div>
);

const Toggle = ({ label, checked, onChange }: { label: string; checked: boolean; onChange: (v: boolean) => void }) => (
  <label className="flex items-center gap-2 cursor-pointer text-sm">
    <input
      type="checkbox"
      checked={checked}
      onChange={(e) => onChange(e.target.checked)}
      className="sr-only"
    />
    <div className={`w-4 h-4 rounded border ${checked ? 'bg-primary border-primary' : 'border-border'} flex items-center justify-center transition-colors`}>
      {checked && <span className="text-accent-foreground text-[10px] font-bold">✓</span>}
    </div>
    <span className="text-foreground/80">{label}</span>
  </label>
);

export default DemandPredictionTab;
