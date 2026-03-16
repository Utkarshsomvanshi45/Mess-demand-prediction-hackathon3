/**
 * api.ts
 * ------
 * Centralized API client for Mess Demand & Food Waste Management System.
 * All calls go to FastAPI backend running at localhost:8000.
 *
 * Usage in any component:
 *   import { api } from '@/lib/api';
 *   const result = await api.predict({ ... });
 */

const BASE_URL = "http://localhost:8000";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(err.detail || `API error ${res.status}`);
  }
  return res.json();
}

// --------------------------------------------------
// REQUEST / RESPONSE TYPES
// --------------------------------------------------

export interface PredictRequest {
  day_of_week:          string;
  meal_type:            string;
  primary_item:         string;
  menu_demand_tier:     string;
  has_paneer:           number;
  has_chicken:          number;
  has_egg:              number;
  has_dessert:          number;
  has_special_cuisine:  number;
  has_drink:            number;
  has_fruit:            number;
  hostel_occupancy_pct: number;
  semester_phase:       string;
  is_weekend:           number;
  previous_meal_demand: string;
}

export interface WasteMetrics {
  recommended: number;
  expected:    number;
  waste:       number;
  wastePct:    number;
  cost:        number;
}

export interface PredictResponse {
  demand:        "High" | "Medium" | "Low";
  waste:         WasteMetrics;
  model_version: number;
}

export interface EDAResponse {
  kpis: {
    total_records:       number;
    avg_occupancy:       number;
    high_demand_count:   number;
    weekend_records:     number;
  };
  demand_dist:    { name: string; value: number; color: string }[];
  demand_by_meal: { meal: string; High: number; Medium: number; Low: number }[];
  occupancy_box:  { level: string; min: number; q1: number; median: number; q3: number; max: number }[];
  phase_demand:   { phase: string; High: number; Medium: number; Low: number }[];
  day_demand:     { day: string; High: number; Medium: number; Low: number }[];
}

export interface WasteStatsResponse {
  kpis: {
    total_waste:      number;
    avg_daily_waste:  number;
    waste_rate:       number;
    cost_of_waste:    number;
  };
  waste_by_meal:  { name: string; portions: number; pct: number }[];
  weekly_trend:   { week: string; waste: number }[];
  accuracy_donut: { name: string; value: number; color: string }[];
  waste_by_phase: { phase: string; breakfast: number; lunch: number; dinner: number }[];
  insights:       string[];
}

export interface DataOverviewResponse {
  total_records:  number;
  total_features: number;
  target_column:  string;
  preview:        Record<string, any>[];
  summary_stats:  { column: string; mean: number; min: number; max: number; std: number }[];
}

export interface ModelInfoResponse {
  model_name:         string;
  version:            number;
  accuracy:           number;
  f1_score:           number;
  precision:          number;
  trained_on_records: number;
  training_date:      string;
  feature_importance: { feature: string; importance: number }[];
  confusion_matrix:   number[][];
  confusion_labels:   string[];
}

export interface ModelHistoryResponse {
  versions: {
    version:       string;
    name:          string;
    records:       number;
    accuracy:      number;
    f1:            number;
    date:          string;
    accuracy_pct:  number;
    f1_pct:        number;
    precision_pct: number;
  }[];
}

export interface PipelineStatusResponse {
  total_records:     number;
  trained_on:        number;
  new_records:       number;
  threshold:         number;
  ready_to_retrain:  boolean;
  last_trained:      string;
  current_version:   number;
  model_name:        string;
}

export interface AddDataResponse {
  success:        boolean;
  records_added:  number;
  total_records:  number;
  message:        string;
}

export interface RetrainResponse {
  success:      boolean;
  message:      string;
  new_version?: number;
  model_file?:  string;
  records_used?: number;
  accuracy?:    number;
  f1?:          number;
  new_records?: number;
  threshold?:   number;
}

// --------------------------------------------------
// API CLIENT
// --------------------------------------------------
export const api = {

  predict: (body: PredictRequest) =>
    request<PredictResponse>("/predict", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  eda: () =>
    request<EDAResponse>("/eda"),

  wasteStats: () =>
    request<WasteStatsResponse>("/waste-stats"),

  dataOverview: () =>
    request<DataOverviewResponse>("/data"),

  modelInfo: () =>
    request<ModelInfoResponse>("/model-info"),

  modelHistory: () =>
    request<ModelHistoryResponse>("/model-history"),

  pipelineStatus: () =>
    request<PipelineStatusResponse>("/pipeline-status"),

  addData: (body: { days: number; semester_phase: string }) =>
    request<AddDataResponse>("/add-data", {
      method: "POST",
      body: JSON.stringify(body),
    }),

  retrain: (threshold = 100) =>
    request<RetrainResponse>("/retrain", {
      method: "POST",
      body: JSON.stringify({ threshold }),
    }),
};
