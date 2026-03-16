import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import DashboardHeader from '@/components/DashboardHeader';
import DemandPredictionTab from '@/components/DemandPredictionTab';
import WasteManagementTab from '@/components/WasteManagementTab';
import EDATab from '@/components/EDATab';
import DataOverviewTab from '@/components/DataOverviewTab';
import ModelInfoTab from '@/components/ModelInfoTab';
import { Sparkles, Trash2, BarChart3, Database, Cpu } from 'lucide-react';

const TABS = [
  { value: 'prediction', label: 'Demand Prediction', icon: Sparkles },
  { value: 'waste', label: 'Waste Management', icon: Trash2 },
  { value: 'eda', label: 'EDA & Insights', icon: BarChart3 },
  { value: 'data', label: 'Data Overview', icon: Database },
  { value: 'model', label: 'Model Info & Pipeline', icon: Cpu },
];

const Dashboard = () => {
  return (
    <div className="min-h-screen bg-background text-foreground p-4 lg:p-6">
      <div className="max-w-[1400px] mx-auto">
        <DashboardHeader />
        <Tabs defaultValue="prediction" className="space-y-6">
          <TabsList className="bg-card border border-border p-1 h-auto flex-wrap gap-1">
            {TABS.map(({ value, label, icon: Icon }) => (
              <TabsTrigger
                key={value}
                value={value}
                className="px-4 py-2.5 text-sm data-[state=active]:bg-primary data-[state=active]:text-accent-foreground rounded-xl transition-all duration-200"
              >
                <Icon className="w-4 h-4 mr-2" />
                {label}
              </TabsTrigger>
            ))}
          </TabsList>

          <TabsContent value="prediction"><DemandPredictionTab /></TabsContent>
          <TabsContent value="waste"><WasteManagementTab /></TabsContent>
          <TabsContent value="eda"><EDATab /></TabsContent>
          <TabsContent value="data"><DataOverviewTab /></TabsContent>
          <TabsContent value="model"><ModelInfoTab /></TabsContent>
        </Tabs>
      </div>
    </div>
  );
};

export default Dashboard;
