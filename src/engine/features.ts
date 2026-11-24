import { mean, std } from './math';

export interface FeatureScaler {
  means: number[];
  stds: number[];
  targetMean: number;
  targetStd: number;
}

export interface SupervisedWindow {
  X: number[][];
  y: number[];
  targetIndex: number[];
}

export function computeLogReturns(closes: readonly number[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    const prev = Math.max(1e-6, closes[i - 1]);
    const curr = Math.max(1e-6, closes[i]);
    returns.push(Math.log(curr / prev));
  }
  return returns;
}

export function extractFeatureVector(
  logReturns: readonly number[],
  returnIdx: number,
  lookback: number,
): number[] {
  const features: number[] = [];

  for (let lag = 1; lag <= lookback; lag++) {
    features.push(logReturns[returnIdx - lag] ?? 0);
  }

  let mom5 = 0;
  for (let k = 1; k <= Math.min(5, returnIdx); k++) mom5 += logReturns[returnIdx - k];
  let mom15 = 0;
  for (let k = 1; k <= Math.min(15, returnIdx); k++) mom15 += logReturns[returnIdx - k];

  const volSlice: number[] = [];
  for (let k = 1; k <= Math.min(10, returnIdx); k++) volSlice.push(logReturns[returnIdx - k]);
  const vol10 = std(volSlice);

  let emaFast = 0;
  let emaSlow = 0;
  const start = Math.max(0, returnIdx - Math.max(lookback, 15));
  for (let i = start; i < returnIdx; i++) {
    emaFast = 0.35 * logReturns[i] + 0.65 * emaFast;
    emaSlow = 0.12 * logReturns[i] + 0.88 * emaSlow;
  }

  features.push(mom5, mom15, vol10, emaFast - emaSlow);
  return features;
}

export function buildSupervisedDataset(
  closes: readonly number[],
  lookback: number,
): SupervisedWindow {
  const logReturns = computeLogReturns(closes);
  const minHistory = Math.max(lookback, 15);
  const X: number[][] = [];
  const y: number[] = [];
  const targetIndex: number[] = [];

  for (let rIdx = minHistory; rIdx < logReturns.length; rIdx++) {
    X.push(extractFeatureVector(logReturns, rIdx, lookback));
    y.push(logReturns[rIdx]);
    targetIndex.push(rIdx + 1);
  }

  return { X, y, targetIndex };
}

export function fitScaler(
  trainX: readonly (readonly number[])[],
  trainY: readonly number[],
): FeatureScaler {
  const numFeatures = trainX[0]?.length ?? 0;
  const means = new Array<number>(numFeatures).fill(0);
  const stds = new Array<number>(numFeatures).fill(1);

  for (let f = 0; f < numFeatures; f++) {
    const col = trainX.map((row) => row[f]);
    const m = mean(col);
    const s = Math.max(1e-5, std(col, m));
    means[f] = m;
    stds[f] = s;
  }

  const targetMean = mean(trainY);
  const targetStd = Math.max(1e-5, std(trainY, targetMean));

  return { means, stds, targetMean, targetStd };
}

export function transformFeatures(x: readonly number[], scaler: FeatureScaler): number[] {
  return x.map((v, idx) => {
    const z = (v - scaler.means[idx]) / scaler.stds[idx];
    return Math.max(-6, Math.min(6, z));
  });
}

export function transformTarget(y: number, scaler: FeatureScaler): number {
  return (y - scaler.targetMean) / scaler.targetStd;
}

export function inverseTransformTarget(yNorm: number, scaler: FeatureScaler): number {
  return yNorm * scaler.targetStd + scaler.targetMean;
}
