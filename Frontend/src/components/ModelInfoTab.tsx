import { useState } from 'react';
import { Cpu, BarChart3, Activity, Award, Database, RefreshCw, CheckCircle2, Loader2 } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, LineChart, Line, Legend } from 'recharts';
import { FEATURE_IMPORTANCE, CONFUSION_MATRIX, PERFORMANCE_HISTORY, MODEL_VERSIONS } from '@/lib/constants';

const tt = {
  contentStyle: { backgroundColor: '#0f1623', border: '1px solid #f59e0b33', borderRadius: '12px', color: '#f1f5f9', fontSize: '12px' },
};

const LABELS = ['Low', 'Medium', 'High'];

const ModelInfoTab = () => {
  const [addDataState, setAddDataState] = useState<'idle' | 'loading' | 'done'>('idle');
  const [retrainState, setRetrainState] = useState<'idle' | 'loading' | 'done'>('idle');
  const [retrainStep, setRetrainStep] = useState(0);
  const [addDays, setAddDays] = useState('14');
  const [addPhase, setAddPhase] = useState('Regular');

  const RETRAIN_STEPS = [
    'Loading data from database...',
    'Preprocessing features...',
    'Training RandomForestClassifier...',
    'Evaluating on held-out test split...',
    'Saving model_v5.pkl...',
    'Updating model registry...',
  ];

  const handleAddData = () => {
    setAddDataState('loading');
    setTimeout(() => setAddDataState('done'), 2000);
  };

  const handleRetrain = () => {
    setRetrainState('loading');
    setRetrainStep(0);
    let step = 0;
    const interval = setInterval(() => {
      step++;
      setRetrainStep(step);
      if (step >= RETRAIN_STEPS.length) {
        clearInterval(interval);
        setTimeout(() => setRetrainState('done'), 500);
      }
    }, 600);
  };

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Model Stats */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {[
          { icon: Cpu, label: 'Algorithm', value: 'RandomForest' },
          { icon: BarChart3, label: 'Version', value: 'v4' },
          { icon: Activity, label: 'Accuracy', value: '91.4%' },
          { icon: Award, label: 'F1 Score', value: '88.2%' },
        ].map(({ icon: Icon, label, value }) => (
          <div key={label} className="card-surface p-5 space-y-2">
            <div className="flex items-center gap-2">
              <Icon className="w-4 h-4 text-primary" />
              <p className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">{label}</p>
            </div>
            <p className="text-2xl font-outfit font-bold text-primary">{value}</p>
          </div>
        ))}
      </div>

      {/* Feature Importance + Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="card-surface p-6">
          <h3 className="font-outfit font-bold mb-4">Feature Importance</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={FEATURE_IMPORTANCE} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
              <XAxis type="number" stroke="#64748b" fontSize={11} domain={[0, 0.5]} />
              <YAxis type="category" dataKey="feature" stroke="#64748b" fontSize={10} width={140} />
              <Tooltip {...tt} />
              <Bar dataKey="importance" fill="#f59e0b" radius={[0, 6, 6, 0]} animationDuration={1500} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card-surface p-6">
          <h3 className="font-outfit font-bold mb-4">Confusion Matrix</h3>
          <div className="flex items-center gap-1 mb-2">
            <span className="text-[10px] text-muted-foreground w-20" />
            {LABELS.map((l) => (
              <span key={l} className="flex-1 text-center text-[10px] text-muted-foreground font-medium">{l}</span>
            ))}
          </div>
          <p className="text-[10px] text-muted-foreground text-center mb-1">← Predicted →</p>
          {CONFUSION_MATRIX.map((row, i) => (
            <div key={i} className="flex items-center gap-1 mb-1">
              <span className="w-20 text-[10px] text-muted-foreground text-right pr-2">{LABELS[i]}</span>
              {row.map((val, j) => {
                const isDiag = i === j;
                return (
                  <div key={j} className={`flex-1 h-16 rounded-lg flex items-center justify-center text-lg font-outfit font-bold transition-colors ${isDiag ? 'bg-primary/20 text-primary border border-primary/30' : 'bg-card text-muted-foreground border border-border'}`}>
                    {val}
                  </div>
                );
              })}
            </div>
          ))}
          <p className="text-[10px] text-muted-foreground mt-2">↑ Actual</p>
        </div>
      </div>

      {/* Performance Evolution */}
      <div className="card-surface p-6">
        <h3 className="font-outfit font-bold mb-4">Model Performance Evolution</h3>
        <ResponsiveContainer width="100%" height={280}>
          <LineChart data={PERFORMANCE_HISTORY}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1e2a3a" />
            <XAxis dataKey="version" stroke="#64748b" fontSize={12} />
            <YAxis stroke="#64748b" fontSize={11} domain={[70, 100]} />
            <Tooltip {...tt} />
            <Line type="monotone" dataKey="accuracy" stroke="#f59e0b" strokeWidth={2} dot={{ r: 5, fill: '#f59e0b' }} animationDuration={1500} />
            <Line type="monotone" dataKey="f1" stroke="#3b82f6" strokeWidth={2} dot={{ r: 5, fill: '#3b82f6' }} animationDuration={1500} />
            <Line type="monotone" dataKey="precision" stroke="#10b981" strokeWidth={2} dot={{ r: 5, fill: '#10b981' }} animationDuration={1500} />
            <Legend wrapperStyle={{ fontSize: '12px' }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Registry */}
      <div className="card-surface p-6">
        <h3 className="font-outfit font-bold mb-4">Model Registry History</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-border">
              {['Version', 'Model Name', 'Records', 'Accuracy', 'F1', 'Training Date'].map((h) => (
                <th key={h} className="px-4 py-2 text-left text-muted-foreground font-medium">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {MODEL_VERSIONS.map((v, i) => (
              <tr key={v.version} className={`border-b border-border/30 ${i % 2 === 1 ? 'bg-background/30' : ''}`}>
                <td className="px-4 py-3 text-primary font-semibold font-mono">{v.version}</td>
                <td className="px-4 py-3">{v.name}</td>
                <td className="px-4 py-3">{v.records.toLocaleString()}</td>
                <td className="px-4 py-3">{v.accuracy}%</td>
                <td className="px-4 py-3">{v.f1}%</td>
                <td className="px-4 py-3 text-muted-foreground">{v.date}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Pipeline Control */}
      <div className="flex items-center gap-4 my-8">
        <div className="flex-1 h-px bg-border" />
        <span className="text-[10px] font-bold text-muted-foreground tracking-[0.2em] uppercase">⚙️ Pipeline Control</span>
        <div className="flex-1 h-px bg-border" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Inject Data */}
        <div className="card-surface glow-hover p-8 relative overflow-hidden">
          <div className="absolute top-4 right-6 text-[10px] font-bold text-muted-foreground tracking-widest uppercase">
            Current Records — 1,020
          </div>
          <div className="flex items-start gap-6">
            <div className="p-4 bg-primary rounded-xl flex-shrink-0">
              <Database className="w-6 h-6 text-accent-foreground" />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-outfit font-bold">Inject New Training Data</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                Generates a new batch of synthetic mess records and appends them to the database. Simulates real-world data accumulation over time.
              </p>
            </div>
          </div>

          <div className="mt-6 grid grid-cols-2 gap-4">
            <div className="space-y-2">
              <label className="text-xs text-muted-foreground">Days of Data</label>
              <select value={addDays} onChange={(e) => setAddDays(e.target.value)} className="w-full bg-background border border-border rounded-xl px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary">
                <option value="7">7 days</option>
                <option value="14">14 days</option>
                <option value="30">30 days</option>
              </select>
            </div>
            <div className="space-y-2">
              <label className="text-xs text-muted-foreground">Semester Phase</label>
              <select value={addPhase} onChange={(e) => setAddPhase(e.target.value)} className="w-full bg-background border border-border rounded-xl px-3 py-2.5 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary">
                <option>Regular</option>
                <option>Exams</option>
                <option>Holidays</option>
              </select>
            </div>
          </div>

          {addDataState === 'idle' && (
            <button onClick={handleAddData} className="btn-primary mt-6 text-sm">➕ Add Data</button>
          )}
          {addDataState === 'loading' && (
            <div className="mt-6 flex items-center gap-3 text-sm text-primary">
              <Loader2 className="w-4 h-4 animate-spin" />
              Generating and inserting records...
            </div>
          )}
          {addDataState === 'done' && (
            <div className="mt-6 flex items-center gap-3 text-sm text-success">
              <CheckCircle2 className="w-4 h-4" />
              +420 records added · Database now has 1,440 records
            </div>
          )}
        </div>

        {/* Retrain */}
        <div className="card-surface glow-hover p-8 relative">
          <div className="absolute top-4 right-6 flex items-center gap-2">
            <span className="text-[10px] font-bold text-muted-foreground tracking-widest uppercase">Ready Status</span>
            <span className="text-[10px] font-bold text-primary bg-primary/10 px-2 py-0.5 rounded">210/100 ✅</span>
          </div>
          <div className="flex items-start gap-6">
            <div className="p-4 bg-primary rounded-xl flex-shrink-0">
              <RefreshCw className="w-6 h-6 text-accent-foreground" />
            </div>
            <div className="space-y-2">
              <h3 className="text-xl font-outfit font-bold">Retrain Prediction Model</h3>
              <p className="text-muted-foreground text-sm leading-relaxed">
                Checks if enough new data exists since last training. If threshold met (100+ new records), retrains and saves new version to registry.
              </p>
            </div>
          </div>

          <div className="mt-6 space-y-3 text-xs">
            {[
              ['Last trained on', '2024-08-10'],
              ['Records used', '1,020'],
              ['New records since', '210'],
              ['Threshold status', 'Met (210 ≥ 100)'],
            ].map(([k, v]) => (
              <div key={k} className="flex justify-between">
                <span className="text-muted-foreground">{k}</span>
                <span className="font-medium">{v}</span>
              </div>
            ))}
          </div>

          <div className="mt-4 space-y-1">
            <div className="h-2 w-full bg-border rounded-full overflow-hidden">
              <div className="h-full bg-primary rounded-full" style={{ width: '100%', boxShadow: '0 0 12px rgba(245,158,11,0.4)' }} />
            </div>
            <p className="text-[10px] text-primary font-semibold">210/100 — Ready</p>
          </div>

          {retrainState === 'idle' && (
            <button onClick={handleRetrain} className="btn-secondary mt-6 text-sm">🔄 Retrain Model</button>
          )}
          {retrainState === 'loading' && (
            <div className="mt-6 space-y-2">
              {RETRAIN_STEPS.map((step, i) => (
                <div key={i} className={`flex items-center gap-2 text-sm transition-opacity duration-300 ${i < retrainStep ? 'opacity-100' : 'opacity-30'}`}>
                  {i < retrainStep ? (
                    <CheckCircle2 className="w-4 h-4 text-success" />
                  ) : (
                    <Loader2 className="w-4 h-4 animate-spin text-primary" />
                  )}
                  <span className={i < retrainStep ? 'text-success' : 'text-muted-foreground'}>{step}</span>
                </div>
              ))}
            </div>
          )}
          {retrainState === 'done' && (
            <div className="mt-6 card-surface p-4 border-primary/30">
              <div className="flex items-center gap-2 text-success text-sm font-semibold mb-2">
                <CheckCircle2 className="w-4 h-4" /> Model v5 saved
              </div>
              <p className="text-xs text-muted-foreground">Accuracy 92.8% · F1 89.3% · +420 records</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ModelInfoTab;
