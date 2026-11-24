import test from 'node:test';
import assert from 'node:assert/strict';
import { MARKET_IDS, buildFallbackMarketSeries } from '../data/marketsCatalog';
import {
  buildBenchmarkCsv,
  runMarketBenchmarkSuite,
  selectChampionModel,
  sortModelResults,
} from './benchmark';
import { runAutoregressiveModel, runHoltWintersModel, runRidgePolyModel } from './classicalModels';
import { buildSupervisedDataset, fitScaler, transformFeatures } from './features';
import { computeMetrics, solveRidgeRegression } from './math';
import { runNeuralModel, sanitizeNeuralConfig } from './neuralNet';

test('solveRidgeRegression recovers exact linear relationship with tiny regularization', () => {
  const X = [
    [1, 0],
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
  ];
  const y = X.map(([, x]) => 5 + 2.5 * x);
  const w = solveRidgeRegression(X, y, 1e-6, false);
  assert.ok(Math.abs(w[0] - 5) < 1e-3, `Expected intercept ~5, got ${w[0]}`);
  assert.ok(Math.abs(w[1] - 2.5) < 1e-3, `Expected slope ~2.5, got ${w[1]}`);
});

test('feature pipeline produces finite zero-centered training features with no leakage', () => {
  const series = buildFallbackMarketSeries('nasdaq');
  const closes = series.bars.map((b) => b.close);
  const ds = buildSupervisedDataset(closes, 10);
  assert.ok(ds.X.length > 200);
  assert.equal(ds.X[0].length, 14); // 10 lags + 4 momentum/vol/ema features

  const split = ds.X.length - 25;
  const scaler = fitScaler(ds.X.slice(0, split), ds.y.slice(0, split));
  const scaledSample = transformFeatures(ds.X[split], scaler);
  for (const val of scaledSample) {
    assert.ok(Number.isFinite(val));
    assert.ok(val >= -6 && val <= 6);
  }
});

test('sanitizeNeuralConfig clamps invalid layer counts, unit widths, and hyperparameters', () => {
  const cleaned = sanitizeNeuralConfig({
    id: 'test-clamp',
    name: '  Over-wide Net  ',
    hiddenLayers: [2, 999, 12, 32], // 4 layers -> capped to 3; units clamped to [4, 64]
    activation: 'silu',
    lookback: 99,
    learningRate: 5,
    epochs: 500,
    l2: -1,
    residualSkip: true,
  });

  assert.equal(cleaned.name, 'Over-wide Net');
  assert.deepEqual(cleaned.hiddenLayers, [4, 64, 12]);
  assert.equal(cleaned.lookback, 30);
  assert.equal(cleaned.learningRate, 0.05);
  assert.equal(cleaned.epochs, 200);
  assert.equal(cleaned.l2, 0);
});

test('neural network trains across 1-3 layers, reduces loss, and produces finite forecasts', () => {
  const series = buildFallbackMarketSeries('nikkei');
  const result = runNeuralModel(
    series.bars,
    {
      id: 'custom-3l-test',
      name: 'Deep 3-Layer Quant',
      hiddenLayers: [20, 12, 8],
      activation: 'relu',
      lookback: 12,
      learningRate: 0.008,
      epochs: 50,
      l2: 0.002,
      residualSkip: true,
      seed: 77,
      isCustom: true,
    },
    25,
    30,
  );

  assert.equal(result.holdoutDates.length, 25);
  assert.equal(result.forecastDates.length, 30);
  assert.equal(result.forecastValues.length, 30);
  assert.ok(result.lossHistory && result.lossHistory.length === 50);

  const firstLoss = result.lossHistory[0].trainLoss;
  const lastLoss = result.lossHistory[result.lossHistory.length - 1].trainLoss;
  assert.ok(lastLoss < firstLoss, `Expected training loss to decrease (${firstLoss} -> ${lastLoss})`);

  for (let i = 0; i < 30; i++) {
    assert.ok(Number.isFinite(result.forecastValues[i]) && result.forecastValues[i] > 0);
    assert.ok(result.band95Low[i] <= result.band80Low[i]);
    assert.ok(result.band80Low[i] <= result.forecastValues[i]);
    assert.ok(result.forecastValues[i] <= result.band80High[i]);
    assert.ok(result.band80High[i] <= result.band95High[i]);
  }
});

test('classical models evaluate cleanly on holdout and project 30 sessions', () => {
  const series = buildFallbackMarketSeries('sse');
  const hw = runHoltWintersModel(series.bars, 25, 30);
  const rp = runRidgePolyModel(series.bars, 25, 30);
  const ar = runAutoregressiveModel(series.bars, 25, 30, 8);

  for (const model of [hw, rp, ar]) {
    assert.equal(model.holdoutPredicted.length, 25);
    assert.equal(model.forecastValues.length, 30);
    assert.ok(Number.isFinite(model.metrics.rmse) && model.metrics.rmse > 0);
    assert.ok(Number.isFinite(model.metrics.mape) && model.metrics.mape > 0);
  }
});

test('full 5-market benchmark retains all competing models and crowns optimal champion per metric', () => {
  for (const marketId of MARKET_IDS) {
    const series = buildFallbackMarketSeries(marketId);
    const results = runMarketBenchmarkSuite(series.bars, [], 25, 30);
    assert.equal(results.length, 6, `Expected 6 preset models for ${marketId}`);

    const byRmse = sortModelResults(results, 'rmse');
    const champRmse = selectChampionModel(results, 'rmse');
    assert.equal(champRmse?.modelId, byRmse[0].modelId);
    assert.ok(byRmse[0].metrics.rmse <= byRmse[byRmse.length - 1].metrics.rmse);

    const byDir = sortModelResults(results, 'directionalAccuracy');
    assert.ok(
      byDir[0].metrics.directionalAccuracy >= byDir[byDir.length - 1].metrics.directionalAccuracy,
    );
  }

  const nasdaqSeries = buildFallbackMarketSeries('nasdaq');
  const nasdaqRes = runMarketBenchmarkSuite(nasdaqSeries.bars, [], 25, 30);
  const csv = buildBenchmarkCsv(nasdaqSeries.meta.name, nasdaqSeries.meta.symbol, nasdaqRes, 30);
  assert.ok(csv.includes('NASDAQ Composite'));
  assert.equal(csv.split('\n').length, 7); // header + 6 models
});

test('computeMetrics calculates accurate directional accuracy and zero-error baseline', () => {
  const m = computeMetrics([100, 102, 101], [100, 102, 101], 99);
  assert.equal(m.rmse, 0);
  assert.equal(m.mape, 0);
  assert.equal(m.directionalAccuracy, 100);
});
