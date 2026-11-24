import type {
  ModelRunResult,
  NeuralConfig,
  PriceBar,
  RankingMetric,
} from '../types/market';
import {
  runAutoregressiveModel,
  runHoltWintersModel,
  runRidgePolyModel,
} from './classicalModels';
import { runNeuralModel } from './neuralNet';

export const PRESET_NEURAL_CONFIGS: readonly NeuralConfig[] = [
  {
    id: 'mlp-compact',
    name: 'Neural Net · Compact',
    hiddenLayers: [16],
    activation: 'tanh',
    lookback: 10,
    learningRate: 0.006,
    epochs: 65,
    l2: 0.015,
    residualSkip: false,
    seed: 101,
    isCustom: false,
  },
  {
    id: 'mlp-deep',
    name: 'Neural Net · Deep 2L',
    hiddenLayers: [24, 12],
    activation: 'tanh',
    lookback: 12,
    learningRate: 0.005,
    epochs: 75,
    l2: 0.018,
    residualSkip: false,
    seed: 202,
    isCustom: false,
  },
  {
    id: 'mlp-residual',
    name: 'Neural Net · Wide Residual',
    hiddenLayers: [32, 16],
    activation: 'silu',
    lookback: 14,
    learningRate: 0.0045,
    epochs: 80,
    l2: 0.022,
    residualSkip: true,
    seed: 303,
    isCustom: false,
  },
];

export function sortModelResults(
  results: readonly ModelRunResult[],
  metric: RankingMetric,
): ModelRunResult[] {
  return [...results].sort((a, b) => {
    if (metric === 'directionalAccuracy') {
      if (b.metrics.directionalAccuracy !== a.metrics.directionalAccuracy) {
        return b.metrics.directionalAccuracy - a.metrics.directionalAccuracy;
      }
      return a.metrics.rmse - b.metrics.rmse;
    }
    if (metric === 'mape') {
      if (a.metrics.mape !== b.metrics.mape) return a.metrics.mape - b.metrics.mape;
      return a.metrics.rmse - b.metrics.rmse;
    }
    // Default: lowest RMSE wins
    if (a.metrics.rmse !== b.metrics.rmse) return a.metrics.rmse - b.metrics.rmse;
    return a.metrics.mape - b.metrics.mape;
  });
}

export function selectChampionModel(
  results: readonly ModelRunResult[],
  metric: RankingMetric,
): ModelRunResult | null {
  if (results.length === 0) return null;
  return sortModelResults(results, metric)[0];
}

export function runMarketBenchmarkSuite(
  bars: readonly PriceBar[],
  customConfigs: readonly NeuralConfig[] = [],
  holdoutSize = 25,
  forecastHorizon = 30,
): ModelRunResult[] {
  const results: ModelRunResult[] = [];

  for (const preset of PRESET_NEURAL_CONFIGS) {
    results.push(runNeuralModel(bars, preset, holdoutSize, forecastHorizon));
  }

  for (const custom of customConfigs) {
    results.push(
      runNeuralModel(bars, { ...custom, isCustom: true }, holdoutSize, forecastHorizon),
    );
  }

  results.push(runHoltWintersModel(bars, holdoutSize, forecastHorizon));
  results.push(runRidgePolyModel(bars, holdoutSize, forecastHorizon));
  results.push(runAutoregressiveModel(bars, holdoutSize, forecastHorizon, 8));

  return results;
}

export function buildBenchmarkCsv(
  marketName: string,
  symbol: string,
  results: readonly ModelRunResult[],
  horizon: number,
): string {
  const header = [
    'Market',
    'Symbol',
    'Model Name',
    'Family',
    'Architecture',
    'RMSE',
    'MAE',
    'MAPE (%)',
    'R2',
    'Directional Accuracy (%)',
    `Target (${horizon}D)`,
    'Trained At',
  ];

  const rows = results.map((r) => {
    const targetVal = r.forecastValues[Math.min(horizon, r.forecastValues.length) - 1] ?? 0;
    return [
      `"${marketName}"`,
      `"${symbol}"`,
      `"${r.modelName.replace(/"/g, '""')}"`,
      r.family,
      `"${r.architectureLabel.replace(/"/g, '""')}"`,
      r.metrics.rmse.toFixed(2),
      r.metrics.mae.toFixed(2),
      r.metrics.mape.toFixed(3),
      r.metrics.r2.toFixed(4),
      r.metrics.directionalAccuracy.toFixed(1),
      targetVal.toFixed(2),
      r.trainedAt,
    ].join(',');
  });

  return [header.join(','), ...rows].join('\n');
}
