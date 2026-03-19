export type MealType = 'Breakfast' | 'Lunch' | 'Dinner';
export type DemandLevel = 'High' | 'Medium' | 'Low';
export type SemesterPhase = 'Regular' | 'Exams' | 'Holidays';

export const DAYS_OF_WEEK = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];

export const DISHES_BY_MEAL: Record<MealType, string[]> = {
  Breakfast: ['Paratha', 'Idli Vada', 'Misal Pav', 'Dhokla', 'Uttapam', 'Sabudana Vada', 'Poha', 'Vada Pav', 'Bombay Sandwich', 'Coleslaw Sandwich', 'Besan Chilla', 'Vermicelli Upma', 'Rava Upma'],
  Lunch: ['Kadhi Pakoda', 'Bhindi Kurkure', 'Chole', 'Baingan Bharta', 'Paneer', 'Soya 65', 'Aloo Jeera', 'Aloo Bhindi', 'Aloo Capsicum', 'Sev Tamatar', 'Methi', 'Baingan Masala', 'Chana Masala', 'Mix Veg', 'Soya Masala', 'Matki', 'Tendli', 'Cabbage', 'Lauki', 'Karela', 'Turai', 'Chawali', 'Rajma', 'Capsicum'],
  Dinner: ['Chicken', 'Paneer', 'Biryani', 'Pav Bhaji', 'Egg', 'Chole', 'Mexican', 'Mix Veg', 'Soya Masala', 'Chinese', 'Tendli', 'Cabbage', 'Lauki', 'Karela', 'Turai', 'Chawali', 'Rajma', 'Capsicum'],
};

const HIGH_TIER = new Set(['Paratha', 'Idli Vada', 'Misal Pav', 'Dhokla', 'Kadhi Pakoda', 'Bhindi Kurkure', 'Chole', 'Baingan Bharta', 'Paneer', 'Soya 65', 'Chicken', 'Biryani', 'Pav Bhaji', 'Egg']);
const MEDIUM_TIER = new Set(['Uttapam', 'Sabudana Vada', 'Poha', 'Vada Pav', 'Aloo Jeera', 'Aloo Bhindi', 'Aloo Capsicum', 'Sev Tamatar', 'Methi', 'Baingan Masala', 'Chana Masala', 'Mix Veg', 'Soya Masala', 'Matki', 'Mexican']);

export const getTier = (dish: string): DemandLevel => {
  if (HIGH_TIER.has(dish)) return 'High';
  if (MEDIUM_TIER.has(dish)) return 'Medium';
  return 'Low';
};

export const MEAL_FACTORS: Record<MealType, number> = { Breakfast: 0.6, Lunch: 0.9, Dinner: 0.75 };
export const DEMAND_FACTORS: Record<DemandLevel, number> = { High: 0.92, Medium: 0.75, Low: 0.55 };

export const calculateDemand = (occupancy: number, tier: DemandLevel): DemandLevel => {
  if (occupancy > 80 && tier === 'High') return 'High';
  if (occupancy < 50) return 'Low';
  return 'Medium';
};

export const getWasteMetrics = (occupancy: number, meal: MealType, demand: DemandLevel) => {
  const recommended = Math.round(occupancy * 3 * MEAL_FACTORS[meal]);
  const expected = Math.round(recommended * DEMAND_FACTORS[demand]);
  const waste = recommended - expected;
  const wastePct = recommended > 0 ? (waste / recommended) * 100 : 0;
  const cost = waste * 45;
  return { recommended, expected, waste, wastePct, cost };
};

export const MOCK_DATASET = Array.from({ length: 10 }, (_, i) => ({
  id: i + 1,
  meal_date: `2024-0${Math.ceil((i + 1) / 3)}-${String(10 + i).padStart(2, '0')}`,
  day_of_week: DAYS_OF_WEEK[i % 7],
  meal_type: (['Breakfast', 'Lunch', 'Dinner'] as MealType[])[i % 3],
  primary_item: ['Paratha', 'Kadhi Pakoda', 'Chicken', 'Poha', 'Chole', 'Biryani', 'Uttapam', 'Mix Veg', 'Paneer', 'Idli Vada'][i],
  menu_demand_tier: (['High', 'High', 'High', 'Medium', 'High', 'High', 'Medium', 'Medium', 'High', 'High'] as DemandLevel[])[i],
  has_paneer: i === 4 || i === 8 ? 1 : 0,
  has_chicken: i === 2 ? 1 : 0,
  has_egg: i === 5 ? 1 : 0,
  has_dessert: i % 3 === 0 ? 1 : 0,
  hostel_occupancy_pct: 60 + Math.round(Math.random() * 35),
  semester_phase: (['Regular', 'Regular', 'Exams', 'Regular', 'Holidays', 'Regular', 'Exams', 'Regular', 'Regular', 'Holidays'] as SemesterPhase[])[i],
  demand_level: (['High', 'Medium', 'High', 'Medium', 'Low', 'High', 'Medium', 'Medium', 'High', 'Low'] as DemandLevel[])[i],
}));

export const MODEL_VERSIONS = [
  { version: 'v1', name: 'RandomForestClassifier', records: 200, accuracy: 78.2, f1: 72.1, date: '2024-01-15' },
  { version: 'v2', name: 'RandomForestClassifier', records: 480, accuracy: 84.5, f1: 80.3, date: '2024-03-02' },
  { version: 'v3', name: 'RandomForestClassifier', records: 750, accuracy: 88.1, f1: 85.4, date: '2024-05-18' },
  { version: 'v4', name: 'RandomForestClassifier', records: 1020, accuracy: 91.4, f1: 88.2, date: '2024-08-10' },
];

export const FEATURE_IMPORTANCE = [
  { feature: 'menu_demand_tier', importance: 0.46 },
  { feature: 'hostel_occupancy_pct', importance: 0.31 },
  { feature: 'has_dessert', importance: 0.22 },
  { feature: 'day_of_week', importance: 0.08 },
  { feature: 'meal_type', importance: 0.06 },
  { feature: 'has_paneer', importance: 0.04 },
  { feature: 'has_chicken', importance: 0.03 },
  { feature: 'semester_phase', importance: 0.03 },
  { feature: 'is_weekend', importance: 0.02 },
];

export const CONFUSION_MATRIX = [
  [142, 12, 3],
  [8, 98, 15],
  [2, 10, 50],
];

export const PERFORMANCE_HISTORY = [
  { version: 'v1', accuracy: 78, f1: 72, precision: 75 },
  { version: 'v2', accuracy: 84, f1: 80, precision: 82 },
  { version: 'v3', accuracy: 88, f1: 85, precision: 87 },
  { version: 'v4', accuracy: 91, f1: 88, precision: 90 },
];
