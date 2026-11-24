export type MarketId = 'nasdaq' | 'sse' | 'kospi' | 'nikkei' | 'hsi';

export type DataSourceKind = 'live' | 'cached' | 'fallback';

export type ForecastHorizon = 7 | 14 | 30;

export type ChartRange = '3M' | '6M' | '1Y';

export type RankingMetric = 'rmse' | 'mape' | 'directionalAccuracy';

export type WorkspaceView = 'forecast' | 'arena' | 'lab' | 'overview';

export type ActivationName = 'tanh' | 'relu' | 'silu';

export interface MarketMeta {
  id: MarketId;
  region: string;
  countryCode: string;
  name: string;
  shortName: string;
  symbol: string;
  exchange: string;
  currency: string;
  currencySymbol: string;
  accentColor: string;
}

export interface PriceBar {
  date: string; // YYYY-MM-DD
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface MarketSeries {
  meta: MarketMeta;
  source: DataSourceKind;
  lastUpdated: string;
  note?: string;
  bars: PriceBar[];
}

export interface NeuralConfig {
  id: string;
  name: string;
  hiddenLayers: number[]; // 1 to 3 hidden layers, each 4..64 units
  activation: ActivationName;
  lookback: number; // 5..30 trading days
  learningRate: number; // 0.0005..0.05
  epochs: number; // 20..200
  l2: number; // 0..0.05
  residualSkip: boolean;
  seed?: number;
  isCustom?: boolean;
}

export interface EpochLossPoint {
  epoch: number;
  trainLoss: number;
  valLoss: number;
}

export interface ModelMetrics {
  rmse: number;
  mae: number;
  mape: number; // percentage, e.g. 1.42 means 1.42%
  r2: number;
  directionalAccuracy: number; // percentage, e.g. 64.0
}

export interface ModelRunResult {
  modelId: string;
  modelName: string;
  family: 'neural' | 'statistical';
  architectureLabel: string;
  isCustom: boolean;
  neuralConfig?: NeuralConfig;
  trainedAt: string;
  trainingMs: number;
  metrics: ModelMetrics;
  holdoutDates: string[];
  holdoutActual: number[];
  holdoutPredicted: number[];
  forecastDates: string[]; // 30 trading days precomputed; sliced for 7D/14D/30D
  forecastValues: number[];
  band80Low: number[];
  band80High: number[];
  band95Low: number[];
  band95High: number[];
  lossHistory?: EpochLossPoint[];
}

export interface BenchmarkSnapshot {
  id: string;
  timestamp: string;
  marketId: MarketId;
  dataSource: DataSourceKind;
  latestClose: number;
  latestDate: string;
  rankingMetric: RankingMetric;
  championModelId: string;
  championModelName: string;
  championRmse: number;
  championMape: number;
  championTarget30d: number;
  modelsEvaluated: number;
}

export interface PersistedWorkspaceState {
  customConfigsByMarket: Record<MarketId, NeuralConfig[]>;
  pinnedOverlaysByMarket: Record<MarketId, string[]>;
  rankingMetric: RankingMetric;
  snapshotHistory: BenchmarkSnapshot[];
}
