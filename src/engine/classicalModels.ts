import type { ModelRunResult, PriceBar } from '../types/market';
import { computeLogReturns } from './features';
import {
  buildConfidenceBands,
  clamp,
  computeMetrics,
  generateFutureTradingDates,
  solveRidgeRegression,
} from './math';

/**
 * 1. Holt-Winters Double Exponential Smoothing with Damped Trend.
 * Hyperparameters (alpha, beta, phi) are grid-tuned strictly on the training slice.
 */
export function runHoltWintersModel(
  bars: readonly PriceBar[],
  holdoutSize = 25,
  forecastHorizon = 30,
): ModelRunResult {
  const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const closes = bars.map((b) => b.close);
  const dates = bars.map((b) => b.date);

  const effectiveHoldout = Math.min(holdoutSize, Math.max(10, Math.floor(closes.length * 0.15)));
  const splitIdx = closes.length - effectiveHoldout;
  const train = closes.slice(0, splitIdx);

  let bestParams = { alpha: 0.45, beta: 0.12, phi: 0.88 };
  let bestTrainErr = Number.POSITIVE_INFINITY;

  const alphas = [0.25, 0.45, 0.65, 0.82];
  const betas = [0.05, 0.12, 0.25];
  const phis = [0.8, 0.9, 0.96];

  for (const alpha of alphas) {
    for (const beta of betas) {
      for (const phi of phis) {
        let level = train[0];
        let trend = train[1] - train[0];
        let sqErr = 0;
        for (let t = 1; t < train.length; t++) {
          const pred = level + phi * trend;
          const err = train[t] - pred;
          sqErr += err * err;
          const nextLevel = alpha * train[t] + (1 - alpha) * pred;
          const nextTrend = beta * (nextLevel - level) + (1 - beta) * phi * trend;
          level = nextLevel;
          trend = nextTrend;
        }
        if (sqErr < bestTrainErr) {
          bestTrainErr = sqErr;
          bestParams = { alpha, beta, phi };
        }
      }
    }
  }

  // Walk through training set to arrive at split state, then one-step forecast on holdout
  let level = closes[0];
  let trend = closes[1] - closes[0];
  const { alpha, beta, phi } = bestParams;

  for (let t = 1; t < splitIdx; t++) {
    const pred = level + phi * trend;
    const nextLevel = alpha * closes[t] + (1 - alpha) * pred;
    const nextTrend = beta * (nextLevel - level) + (1 - beta) * phi * trend;
    level = nextLevel;
    trend = nextTrend;
  }

  const holdoutDates: string[] = [];
  const holdoutActual: number[] = [];
  const holdoutPredicted: number[] = [];

  for (let t = splitIdx; t < closes.length; t++) {
    const pred = level + phi * trend;
    holdoutDates.push(dates[t]);
    holdoutActual.push(closes[t]);
    holdoutPredicted.push(Number(Math.max(1, pred).toFixed(2)));

    const nextLevel = alpha * closes[t] + (1 - alpha) * pred;
    const nextTrend = beta * (nextLevel - level) + (1 - beta) * phi * trend;
    level = nextLevel;
    trend = nextTrend;
  }

  const metrics = computeMetrics(holdoutActual, holdoutPredicted, closes[splitIdx - 1]);

  // Multi-step damped trend forecast from final filtered state
  const forecastDates = generateFutureTradingDates(dates[dates.length - 1], forecastHorizon);
  const forecastValues: number[] = [];
  let dampedSum = 0;
  for (let h = 1; h <= forecastHorizon; h++) {
    dampedSum += Math.pow(phi, h);
    const val = Math.max(1, level + dampedSum * trend);
    forecastValues.push(Number(val.toFixed(2)));
  }

  const bands = buildConfidenceBands(forecastValues, holdoutActual, holdoutPredicted);
  const endedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();

  return {
    modelId: 'holt-winters',
    modelName: 'Holt-Winters Damped',
    family: 'statistical',
    architectureLabel: `DES α=${alpha} β=${beta} φ=${phi}`,
    isCustom: false,
    trainedAt: new Date().toISOString(),
    trainingMs: Math.max(1, Math.round(endedAt - startedAt)),
    metrics,
    holdoutDates,
    holdoutActual,
    holdoutPredicted,
    forecastDates,
    forecastValues,
    ...bands,
  };
}

/**
 * 2. Ridge Cubic Trend + AR(3) Residual Correction in log-price space.
 */
export function runRidgePolyModel(
  bars: readonly PriceBar[],
  holdoutSize = 25,
  forecastHorizon = 30,
): ModelRunResult {
  const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const closes = bars.map((b) => b.close);
  const dates = bars.map((b) => b.date);
  const logPrices = closes.map((c) => Math.log(Math.max(1e-6, c)));

  const effectiveHoldout = Math.min(holdoutSize, Math.max(10, Math.floor(closes.length * 0.15)));
  const splitIdx = closes.length - effectiveHoldout;
  const scaleT = Math.max(1, splitIdx - 1);

  const polyFeatures = (t: number) => {
    const u = (t / scaleT) * 2 - 1; // [-1, 1] on training window
    return [1, u, u * u, u * u * u];
  };

  const trainPolyX = Array.from({ length: splitIdx }, (_, t) => polyFeatures(t));
  const trainLogY = logPrices.slice(0, splitIdx);
  const polyWeights = solveRidgeRegression(trainPolyX, trainLogY, 1.8, false);

  const trendAt = (t: number) => {
    const f = polyFeatures(t);
    return f[0] * polyWeights[0] + f[1] * polyWeights[1] + f[2] * polyWeights[2] + f[3] * polyWeights[3];
  };

  // Fit AR(3) on training residuals
  const residuals: number[] = [];
  for (let t = 0; t < splitIdx; t++) {
    residuals.push(logPrices[t] - trendAt(t));
  }

  const arX: number[][] = [];
  const arY: number[] = [];
  for (let t = 3; t < residuals.length; t++) {
    arX.push([1, residuals[t - 1], residuals[t - 2], residuals[t - 3]]);
    arY.push(residuals[t]);
  }
  const arW = solveRidgeRegression(arX, arY, 0.35, false);

  const holdoutDates: string[] = [];
  const holdoutActual: number[] = [];
  const holdoutPredicted: number[] = [];
  const fullResiduals = residuals.slice();

  for (let t = splitIdx; t < closes.length; t++) {
    const rLen = fullResiduals.length;
    const predRes =
      arW[0] +
      arW[1] * fullResiduals[rLen - 1] +
      arW[2] * fullResiduals[rLen - 2] +
      arW[3] * fullResiduals[rLen - 3];
    const predLog = trendAt(t) + clamp(predRes, -0.15, 0.15);
    const predPrice = Math.exp(predLog);

    holdoutDates.push(dates[t]);
    holdoutActual.push(closes[t]);
    holdoutPredicted.push(Number(predPrice.toFixed(2)));
    fullResiduals.push(logPrices[t] - trendAt(t));
  }

  const metrics = computeMetrics(holdoutActual, holdoutPredicted, closes[splitIdx - 1]);

  // Refit smoothly on full window for forward forecast (with damped linear extrapolation past u=1)
  const fullScale = Math.max(1, closes.length - 1);
  const fullPolyX = Array.from({ length: closes.length }, (_, t) => {
    const u = (t / fullScale) * 2 - 1;
    return [1, u, u * u, u * u * u];
  });
  const fullPolyW = solveRidgeRegression(fullPolyX, logPrices, 2.2, false);
  const slopeAtEnd = (fullPolyW[1] + 2 * fullPolyW[2] + 3 * fullPolyW[3]) * (2 / fullScale);
  const clampedSlope = clamp(slopeAtEnd, -0.0022, 0.0022);
  const endTrend = fullPolyW[0] + fullPolyW[1] + fullPolyW[2] + fullPolyW[3];

  const simRes = fullResiduals.slice(-3);
  const forecastDates = generateFutureTradingDates(dates[dates.length - 1], forecastHorizon);
  const forecastValues: number[] = [];

  for (let h = 1; h <= forecastHorizon; h++) {
    const nextRes = clamp(
      (arW[0] + arW[1] * simRes[simRes.length - 1] + arW[2] * simRes[simRes.length - 2] + arW[3] * simRes[simRes.length - 3]) *
        Math.pow(0.95, h),
      -0.12,
      0.12,
    );
    const nextLog = endTrend + clampedSlope * h * Math.pow(0.985, h) + nextRes;
    forecastValues.push(Number(Math.exp(nextLog).toFixed(2)));
    simRes.push(nextRes);
  }

  const bands = buildConfidenceBands(forecastValues, holdoutActual, holdoutPredicted);
  const endedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();

  return {
    modelId: 'ridge-poly',
    modelName: 'Ridge Cubic + AR(3)',
    family: 'statistical',
    architectureLabel: 'Poly(d=3) + AR(3) · λ=1.8',
    isCustom: false,
    trainedAt: new Date().toISOString(),
    trainingMs: Math.max(1, Math.round(endedAt - startedAt)),
    metrics,
    holdoutDates,
    holdoutActual,
    holdoutPredicted,
    forecastDates,
    forecastValues,
    ...bands,
  };
}

/**
 * 3. Autoregressive AR(8) Return Momentum Baseline.
 */
export function runAutoregressiveModel(
  bars: readonly PriceBar[],
  holdoutSize = 25,
  forecastHorizon = 30,
  lags = 8,
): ModelRunResult {
  const startedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const closes = bars.map((b) => b.close);
  const dates = bars.map((b) => b.date);
  const returns = computeLogReturns(closes);

  const X: number[][] = [];
  const y: number[] = [];
  const targetIdx: number[] = [];

  for (let r = lags; r < returns.length; r++) {
    const row = [1];
    for (let k = 1; k <= lags; k++) row.push(returns[r - k]);
    X.push(row);
    y.push(returns[r]);
    targetIdx.push(r + 1);
  }

  const effectiveHoldout = Math.min(holdoutSize, Math.max(10, Math.floor(X.length * 0.15)));
  const splitIdx = X.length - effectiveHoldout;

  const weights = solveRidgeRegression(X.slice(0, splitIdx), y.slice(0, splitIdx), 0.15, false);

  const holdoutDates: string[] = [];
  const holdoutActual: number[] = [];
  const holdoutPredicted: number[] = [];

  for (let i = splitIdx; i < X.length; i++) {
    const day = targetIdx[i];
    const row = X[i];
    let predRet = 0;
    for (let j = 0; j < weights.length; j++) predRet += weights[j] * row[j];
    predRet = clamp(predRet, -0.045, 0.045);

    holdoutDates.push(dates[day]);
    holdoutActual.push(closes[day]);
    holdoutPredicted.push(Number((closes[day - 1] * Math.exp(predRet)).toFixed(2)));
  }

  const metrics = computeMetrics(holdoutActual, holdoutPredicted, closes[targetIdx[splitIdx] - 1]);

  // Forward iterative rollout
  const rolling = returns.slice();
  const forecastDates = generateFutureTradingDates(dates[dates.length - 1], forecastHorizon);
  const forecastValues: number[] = [];
  let runningPrice = closes[closes.length - 1];

  for (let h = 0; h < forecastHorizon; h++) {
    let predRet = weights[0];
    for (let k = 1; k <= lags; k++) {
      predRet += weights[k] * rolling[rolling.length - k];
    }
    predRet = clamp(predRet, -0.028, 0.028);
    runningPrice *= Math.exp(predRet);
    forecastValues.push(Number(runningPrice.toFixed(2)));
    rolling.push(predRet);
  }

  const bands = buildConfidenceBands(forecastValues, holdoutActual, holdoutPredicted);
  const endedAt = typeof performance !== 'undefined' ? performance.now() : Date.now();

  return {
    modelId: 'ar-momentum',
    modelName: `Autoregressive AR(${lags})`,
    family: 'statistical',
    architectureLabel: `Linear AR(${lags}) · Ridge λ=0.15`,
    isCustom: false,
    trainedAt: new Date().toISOString(),
    trainingMs: Math.max(1, Math.round(endedAt - startedAt)),
    metrics,
    holdoutDates,
    holdoutActual,
    holdoutPredicted,
    forecastDates,
    forecastValues,
    ...bands,
  };
}
